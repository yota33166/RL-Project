#!/usr/bin/env python3
import atexit
import gc
import signal
import sys
import tempfile
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing, nn
from torch.multiprocessing import freeze_support
from torchrl._utils import _CKPT_BACKEND
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
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
    check_env_specs,
    set_exploration_type,
)
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.trainers import (
    ClearCudaCache,
    LogScalar,
    LogValidationReward,
    ReplayBufferTrainer,
    Trainer,
    TrainerHookBase,
    UpdateWeights,
)

warnings.filterwarnings("ignore")

SEED = 42

""" Define Hyperparameters """
mp_context = multiprocessing.get_start_method()
is_fork = mp_context == "fork"
device = torch.device("cuda:0" if torch.cuda.is_available() and not is_fork else "cpu")

num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
wd = 0.0  # weight decay
max_grad_norm = 1.0

""" Data collection parameters """
sub_batch_size = 64  # 512
frames_per_batch = 1024  # 2048
# For a complete training, bring the number of frames up to 1M
total_frames = frames_per_batch * 1000  # total number of frames to collect
buffer_size = min(409600, total_frames)  # size of the replay buffer

""" PPO parameters """
# cardinality of the sub-samples gathered from the current data in the inner loop
n_optim = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

env_name = "InvertedDoublePendulum-v4"


class SaveBestValidationReward(LogValidationReward):
    def __init__(
        self,
        model_dir: str,
        value_module: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dir = Path(model_dir)
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
                self.model_dir / f"model_{str(self.best_reward.item())}.pt",
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


def env_maker(env_name, device, render_mode=None):
    env = GymEnv(
        env_name=env_name,
        from_pixels=False,
        pixels_only=False,
        device=device,
        render_mode=render_mode,
    )
    return env


def make_env(
    parallel=False,
    obs_norm_sd=None,
    num_workers=1,
    maker=env_maker,
    render_mode=None,
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}

    obs_norm_kwargs = obs_norm_sd.copy()

    if parallel:
        env_kwargs = {
            "env_name": env_name,
            "device": device,
            "render_mode": render_mode,
        }
        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker, env_kwargs),
            serial_for_single=True,
            mp_start_method=mp_context,
        )
    else:
        base_env = env_maker(env_name, device, render_mode=render_mode)
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            StepCounter(),
            DoubleToFloat(),
            RewardScaling(loc=0.0, scale=1.0),
            ObservationNorm(in_keys=["observation"], **obs_norm_kwargs),
        ),
    )

    return env


def get_norm_stats():
    test_env = make_env()
    test_env.transform[-1].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    obs_norm_sd = test_env.transform[-1].state_dict()
    test_env.close()
    del test_env
    return obs_norm_sd


def make_ppo_model(dummy_env):
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


def load_and_demo_model(model_path):
    # モデルデータ読み込み
    ckpt = torch.load(model_path, map_location=device)

    obs_norm = ckpt["obsnorm_state_dict"]
    env = make_env(render_mode="human", obs_norm_sd=obs_norm)
    policy_module, value_module = make_ppo_model(env)
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
        env.rollout(
            10000, policy_module, break_when_any_done=True
        )  # 10000 ステップのロールアウトを実行
    env.close()
    del env, policy_module, value_module, obs_norm


""" Replay buffer """


def get_replay_buffer(buffer_size, n_optim, batch_size, device):
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(max_size=buffer_size, scratch_dir=buffer_scratch_dir),
        prefetch=n_optim,
        sampler=SamplerWithoutReplacement(),
        transform=lambda td: td.to(device, non_blocking=True),
    )
    return replay_buffer


""" Data collector"""


def get_collector(
    stats,
    num_collectors,
    policy,
    value_module,
    frames_per_batch,
    total_frames,
    device,
):
    if is_fork:
        collector_cls = SyncDataCollector
        env_arg = make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
    else:
        collector_cls = MultiaSyncDataCollector
        env_arg = [
            make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
            for _ in range(num_collectors)
        ]

    data_collector = collector_cls(
        env_arg,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        exploration_type=ExplorationType.RANDOM,
        device=device,
        storing_device=device,
        split_trajs=False,
    )
    return data_collector


def get_loss_module(
    actor_network, critic_network, clip_epsilon=clip_epsilon, entropy_eps=entropy_eps
):
    loss_module = ClipPPOLoss(
        actor_network=actor_network,
        critic_network=critic_network,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
        normalize_advantage=True,
    )
    return loss_module


def cleanup_resources():
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
            elif isinstance(collector, MultiaSyncDataCollector):
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


def _signal_handler(signum, frame):
    print(f"Signal {signum} received. Cleaning up resources...")
    cleanup_resources()
    sys.exit(0)


if __name__ == "__main__":
    freeze_support()  # Windows の場合、マルチプロセスのために必要
    num_workers = 4
    # num_workers = multiprocessing.cpu_count()  # number of workers for parallel envs
    print(f"Number of workers: {num_workers}")
    num_collectors = num_workers  # number of collectors for parallel envs
    load_model = True
    if load_model:
        # RL-Project\models\Ant-v4_Apr30_16-50-57\best_model.pt
        # RL-Project\models\ppo_Ant-v4_May07_16-00-19\model_tensor(0.9754).pt
        # model_data.pthRL-Project\models\
        model_path = Path("models") / "model_1167.pt"
        load_and_demo_model(model_path)
        cleanup_resources()
        sys.exit(0)

    exp_name = f"{env_name}_{uuid.uuid4().hex[:6]}"
    buffer_scratch_dir = tempfile.TemporaryDirectory().name
    save_dir = Path(f"ppo_{env_name}_{datetime.now().strftime('%b%d_%H-%M-%S')}")
    log_dir = Path("logs") / save_dir
    model_dir = Path("models") / save_dir

    if _CKPT_BACKEND == "torchsnapshot":
        save_trainer_file = (
            model_dir / f"trainer_{datetime.now().strftime('%b%d_%H-%M-%S')}",
        )
    elif _CKPT_BACKEND == "torch":
        save_trainer_file = (
            model_dir / f"trainer_{datetime.now().strftime('%b%d_%H-%M-%S')}.pt"
        )
    else:
        raise ValueError(f"Unknown checkpoint backend: {_CKPT_BACKEND}")

    logger = TensorboardLogger(exp_name=exp_name, log_dir=log_dir)
    log_interval = 1  # log interval for Tensorboard

    stats = get_norm_stats()
    test_env = make_env(obs_norm_sd=stats)
    policy_module, value_module = make_ppo_model(test_env)
    print("Running policy:", policy_module(test_env.reset()))
    print("Running value:", value_module(test_env.reset()))

    loss_module = get_loss_module(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_eps=entropy_eps,
    )
    collector = get_collector(
        stats=stats,
        num_collectors=num_collectors,
        policy=policy_module,
        value_module=value_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    trainer = Trainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=1,
        loss_module=loss_module,
        optimizer=optim,
        logger=logger,
        optim_steps_per_batch=n_optim,
        log_interval=log_interval,
        clip_norm=max_grad_norm,
        seed=SEED,
    )
    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_module,
        average_gae=True,
        device=device,
    )

    trainer.register_op("batch_process", advantage_module)
    # batch_subsampler = BatchSubSampler(
    #     batch_size=sub_batch_size,
    # )
    # batch_subsampler.register(trainer)

    buffer_hook = ReplayBufferTrainer(
        # PPOはオンポリシーアルゴリズムなので、buffer_sizeはframes_per_batchと同じ
        get_replay_buffer(
            frames_per_batch, n_optim, batch_size=sub_batch_size, device=device
        ),
        flatten_tensordicts=True,
    )
    buffer_hook.register(trainer)

    weight_updater = UpdateWeights(collector=collector, update_weights_interval=1)
    weight_updater.register(trainer)
    recorder = SaveBestValidationReward(
        model_dir=model_dir,
        value_module=value_module,
        record_interval=10,  # log every 100 optimization steps
        record_frames=1000,  # maximum number of frames in the record
        frame_skip=1,
        policy_exploration=policy_module,
        environment=test_env,
        exploration_type=ExplorationType.DETERMINISTIC,
        log_keys=[("next", "reward")],
        out_keys={("next", "reward"): "r_evaluation"},
        log_pbar=True,
    )
    recorder.register(trainer)

    trainer.register_op("post_optim", scheduler.step)

    log_reward = LogScalar(log_pbar=True)
    log_reward.register(trainer)

    clear_cuda = ClearCudaCache(100)
    trainer.register_op("pre_optim_steps", clear_cuda)

    atexit.register(cleanup_resources)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    """ Training"""
    try:
        trainer.train()
        print("\nTraining completed successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        cleanup_resources()
