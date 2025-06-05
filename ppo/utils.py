#!/usr/bin/env python3
import gc
import os
import sys
import tempfile
import time
import warnings
from typing import Any, Dict, Optional

import pandas as pd
import psutil
import torch
from config import Config
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data import LazyMemmapStorage
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    EnvCreator,
    ObservationNorm,
    ParallelEnv,
    RewardScaling,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import (
    ExplorationType,
    TensorDict,
    TensorDictBase,
    set_exploration_type,
)
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.trainers import LogValidationReward, Trainer
from torchrl.trainers.trainers import TrainerHookBase

warnings.filterwarnings("ignore")


class SaveBestValidationReward(LogValidationReward):
    """報酬の良いモデルを保存するトレーナーフック
    このクラスは、評価時の報酬がこれまでの最良値を更新した場合に、
    モデルと正規化統計を保存します。

    Args:
        LogValidationReward (_type_): _description_
        cfg (Config): 設定オブジェクト
        value_module (torch.nn.Module): 価値関数モジュール.保存用にのみ参照される
    """

    def __init__(
        self,
        cfg: Config,
        value_module: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dir = cfg.log.model_dir
        self.value_module = value_module

        self.best_reward = -float("inf")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def __call__(self, batch: TensorDictBase) -> Dict:
        metrics = super().__call__(batch)
        if metrics is None:
            return None
        reward = metrics.get("total_r_evaluation")
        if reward is not None and reward > self.best_reward:
            self.best_reward = reward
            obs_norm = self.environment.transform[-1]
            # モデルと正規化統計を保存
            torch.save(
                {
                    "policy_state_dict": self.policy_exploration.state_dict(),
                    "value_state_dict": self.value_module.state_dict(),
                    "obsnorm_state_dict": obs_norm.state_dict(),
                    "reward": self.best_reward,
                },
                self.model_dir / f"best_model_{str(int(self.best_reward.item()))}.pt",
            )
            self.trainer.logger.log_scalar("best_model_saved_at", self.best_reward)

        return metrics

    def state_dict(self):
        base = super().state_dict()
        base.update({"best_reward": self.best_reward})
        return base

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.best_reward = state_dict.get("best_reward", -float("inf"))

    def register(self, trainer: Trainer, name: str = "recorder"):
        # Trainer インスタンスを保持
        self.trainer = trainer
        # モジュールとして登録
        trainer.register_module(name, self)
        # post_steps フックとして登録
        trainer.register_op("post_steps_log", self)


class LogLearningRate(TrainerHookBase):
    """学習率をログに記録するトレーナーフック
    このクラスは、最適化器の学習率を定期的にログに記録します。

    Args:
        optimizer (torch.optim.Optimizer): 学習率をログに記録する最適化器
        logger (Logger): ログを記録するロガー
        log_interval (int): 学習率をログに記録する間隔
        name (str): ログの名前（デフォルトは "learning_rate"）
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        logger: TensorboardLogger,
        log_interval: int = 1,
        name: str = "learning_rate",
    ):
        super().__init__()
        self.optimizer = optimizer
        self.logger = logger
        self.log_interval = log_interval
        self.name = name
        self._step = 0

    def __call__(self, *args, **kwargs):
        self._step += 1
        if self._step % self.log_interval == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log_scalar(self.name, lr, step=self._step)

    def register(self, trainer, name="log_lr"):
        trainer.register_op("post_optim", self, name=name)


def env_maker(
    cfg: Config, device: torch.device, render_mode: Optional[str] = None
) -> GymEnv:
    """環境を作成する関数
    この関数は、指定された設定に基づいてGym環境を作成します。
    もし `render_mode` が指定されていれば、レンダリングモードを設定します。
    Args:
        cfg (Config): 設定オブジェクト
        device (torch.device): 使用するデバイス
        render_mode (str, optional): レンダリングモード. デフォルトは None.
    Returns:
        GymEnv: 作成された環境
    """

    env = GymEnv(
        env_name=cfg.env.env_name,
        from_pixels=False,
        pixels_only=False,
        device=device,
        render_mode=render_mode,
    )
    return env


def make_env(
    cfg: Config,
    device: torch.device,
    mp_context: Optional[str] = None,
    parallel: bool = False,
    obs_norm_sd: Optional[Dict[str, Any]] = None,
    maker=env_maker,
    render_mode: Optional[str] = None,
) -> TransformedEnv:
    """環境を変換して作成する関数
    この関数は、指定された設定に基づいて環境を作成し、観測の正規化や報酬のスケーリングなどの変換を適用します。
    Args:
        cfg (Config): 設定オブジェクト
        device (torch.device): 使用するデバイス
        mp_context (str, optional): マルチプロセスのコンテキスト. デフォルトは None.
        parallel (bool, optional): 並列環境を使用するかどうか. デフォルトは False.
        obs_norm_sd (Dict[str, Any], optional): 観測の正規化に使用する標準偏差. デフォルトは None.
        maker (callable, optional): 環境を作成する関数. デフォルトは env_maker.
        render_mode (str, optional): レンダリングモード. デフォルトは None.
    Returns:
        TransformedEnv: 作成された環境
    """
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}

    obs_norm_kwargs = obs_norm_sd.copy()

    if parallel:
        env_kwargs = {
            "cfg": cfg,
            "device": device,
            "render_mode": render_mode,
        }
        base_env = ParallelEnv(
            cfg.env.num_workers,
            EnvCreator(maker, env_kwargs),
            serial_for_single=True,
            mp_start_method=mp_context,
        )
    else:
        base_env = env_maker(cfg, device, render_mode=render_mode)
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            StepCounter(),
            DoubleToFloat(),
            RewardScaling(loc=0.0, scale=cfg.env.reward_scale),
            ObservationNorm(in_keys=["observation"], **obs_norm_kwargs),
        ),
    )

    return env


def get_norm_stats(test_env: TransformedEnv) -> Dict[str, Any]:
    test_env.transform[-1].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    obs_norm_sd = test_env.transform[-1].state_dict()
    test_env.close()
    del test_env
    return obs_norm_sd


# TODO: MLPモジュールつかって上手いことやる（2025/05/21）
def make_ppo_model(
    num_cells: int, dummy_env: GymEnv, device: torch.device
) -> tuple[ProbabilisticActor, ValueOperator]:
    """PPOモデルを作成する関数
    この関数は、指定されたセル数に基づいてPPOモデルを構築します。
    Args:
        num_cells (int): ネットワークのセル数
        dummy_env (GymEnv): ダミー環境
        device (torch.device): 使用するデバイス
    Returns:
        tuple[ProbabilisticActor, ValueOperator]: 作成されたポリシーモジュールと価値モジュール
    """
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * dummy_env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=dummy_env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": dummy_env.action_spec.space.low,
            "high": dummy_env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return policy_module, value_module


def load_and_demo_model(
    cfg: Config, model_path: str, device: torch.device, iteration: int = 1
):
    """モデルを読み込み、デモを実行する関数
    この関数は、指定されたモデルパスからモデルを読み込み、
    環境でデモを実行します。デモの結果は、観測とアクションをCSVファイルに保存します。
    Args:
        cfg (Config): 設定オブジェクト
        model_path (str): モデルのパス
        device (torch.device): 使用するデバイス
        iteration (int, optional): デモの反復回数. デフォルトは 1.
    """
    # モデルデータ読み込み
    ckpt = torch.load(model_path, map_location=device)

    obs_norm = ckpt["obsnorm_state_dict"]
    env = make_env(cfg, device=device, obs_norm_sd=obs_norm, render_mode="human")
    policy_module, value_module = make_ppo_model(cfg.optim.num_cells, env, device)
    policy_module.load_state_dict(ckpt["policy_state_dict"])
    value_module.load_state_dict(ckpt["value_state_dict"])
    # optim.load_state_dict(ckpt["optimizer_state_dict"])
    # scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    # current_frame = ckpt.get("frame", 0)  # フレーム数の復元

    # 評価モードに切り替え
    policy_module.eval()
    value_module.eval()
    env.transform[-1].eval()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(iteration):
            rollout_tensordict = env.rollout(
                1000, policy_module, break_when_any_done=True
            )  # 10000 ステップのロールアウトを実行
            print(f"obs: {rollout_tensordict['observation']}")
            normalized_obs = TensorDict(
                {"observation": rollout_tensordict["observation"]},
                batch_size=[rollout_tensordict["observation"].shape[0]],
            )

            original_obs = env.transform[-1].inv(normalized_obs)
            original_obs = original_obs["observation"]
            obs_array = original_obs.cpu().numpy()
            df_obs = pd.DataFrame(obs_array)
            df_obs.to_csv("observation.csv", index=False, header=False)

            action = rollout_tensordict["action"]
            action_array = action.cpu().numpy()
            df_action = pd.DataFrame(action_array)
            df_action.to_csv("action.csv", index=False, header=False)
    env.close()
    del env, policy_module, value_module, obs_norm


def get_replay_buffer(
    buffer_size: int, prefetch: int, batch_size: int, device: torch.device
) -> TensorDictReplayBuffer:
    """リプレイバッファを取得する関数
    この関数は、指定されたバッファサイズとバッチサイズに基づいてリプレイバッファを作成します。
    Args:
        buffer_size (int): リプレイバッファの最大サイズ
        prefetch (int): あらかじめ読み込むサンプル数
        batch_size (int): バッチサイズ
        device (torch.device): 使用するデバイス
    Returns:
        TensorDictReplayBuffer: 作成されたリプレイバッファ
    """
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(max_size=buffer_size, scratch_dir=tempfile.mkdtemp()),
        prefetch=prefetch,
        sampler=SamplerWithoutReplacement(),
        transform=lambda td: td.to(device, non_blocking=True),
    )
    return replay_buffer


def get_collector(
    cfg: Config,
    mp_context: str,
    stats: Dict[str, Any],
    policy: nn.Module,
    device: torch.device,
):
    """データコレクターを取得する関数
    この関数は、指定された設定に基づいてデータコレクターを作成します。
    Args:
        cfg (Config): 設定オブジェクト
        mp_context (str): マルチプロセスのコンテキスト
        stats (Dict[str, Any]): 観測の正規化に使用する統計情報
        policy (nn.Module): ポリシーモジュール
        device (torch.device): 使用するデバイス
    Returns:
        SyncDataCollector or MultiSyncDataCollector: 作成されたデータコレクター
    """
    if mp_context == "fork":
        collector_cls = SyncDataCollector
        env_arg = make_env(cfg, device, parallel=True, obs_norm_sd=stats)
        print("Using SyncDataCollector")
    else:
        collector_cls = MultiSyncDataCollector
        env_arg = [
            make_env(cfg, device, parallel=True, obs_norm_sd=stats)
            for _ in range(cfg.collector.env_per_collector)
        ]
        print("Using MultiSyncDataCollector")

    data_collector = collector_cls(
        env_arg,
        policy=policy,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        exploration_type=ExplorationType.RANDOM,
        device=device,
        storing_device=device,
        split_trajs=False,
    )
    return data_collector


def get_loss_module(cfg: Config, actor_network: nn.Module, critic_network: nn.Module):
    """損失モジュールを取得する関数
    この関数は、指定された設定に基づいてPPOの損失モジュールを作成します。
    Args:
        cfg (Config): 設定オブジェクト
        actor_network (nn.Module): アクターネットワーク
        critic_network (nn.Module): クリティックネットワーク
    Returns:
        ClipPPOLoss: 作成された損失モジュール
    """
    loss_module = ClipPPOLoss(
        actor_network=actor_network,
        critic_network=critic_network,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_bonus=bool(cfg.loss.entropy_eps),
        entropy_coef=cfg.loss.entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=cfg.loss.critic_coef,
        loss_critic_type="smooth_l1",
        normalize_advantage=True,
    )
    return loss_module


def cleanup_resources():
    """リソースをクリーンアップする関数
    この関数は、トレーナー、コレクター、およびテスト環境のリソースを解放します。
    例外が発生した場合は、エラーメッセージを表示します。
    """
    global trainer, collector, test_env

    print("Cleaning up resources...")
    try:
        if "trainer" in globals() and trainer is not None:
            print("Saving trainer...")
            trainer.save_trainer(True)
    except Exception as e:
        print(f"Failed to save trainer: {e}")

    try:
        if "collector" in globals() and collector is not None:
            print("Shutting down collector...")
            if isinstance(collector, SyncDataCollector):
                print("Shutting down collectors...")
                collector.shutdown()
            elif isinstance(collector, MultiSyncDataCollector):
                print(f"Shutting down {len(collector.collectors)} sub-collectors...")
                for c in collector.collectors:
                    c.shutdown()
    except Exception as e:
        print(f"Failed to shutdown collector: {e}")

    try:
        if "test_env" in globals() and test_env is not None:
            test_env.close()
            if isinstance(test_env, ParallelEnv):
                if hasattr(test_env, "base_env"):
                    print("Closing base environment...")
                    test_env.base_env.close()

    except Exception as e:
        print(f"Failed to close environment: {e}")

    try:
        del trainer, collector, test_env
    except Exception as e:
        print(f"Failed to delete resources: {e}")

    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Cleaning up CUDA device {i}...")
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    print("Resources cleaned up.")


def resource_monitor(cpu_thresh=90, mem_thresh=90, interval=5):
    while True:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        if cpu > cpu_thresh or mem > mem_thresh:
            print(f"[WARN] Exiting: CPU={cpu}%, MEM={mem}%")
            os._exit(1)  # 即時終了（safeではないが確実）
        time.sleep(interval)


def _signal_handler(signum, frame):
    """シグナルハンドラー
    この関数は、SIGINTやSIGTERMシグナルを受け取ったときに呼び出され、
    リソースをクリーンアップし、プログラムを終了します。
    Args:
        signum (int): 受け取ったシグナル番号
        frame (frame): 現在のスタックフレーム
    """
    print(f"Signal {signum} received. Cleaning up resources...")
    cleanup_resources()
    sys.exit(0)
