import os
import tempfile
import uuid
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
from IPython import get_ipython
from tensordict.nn import TensorDictSequential
from torch import multiprocessing, nn
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
    EnvBase,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import DuelingCnnDQNet, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers.csv import CSVLogger
from torchrl.trainers import (
    LogScalar,
    LogValidationReward,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)

# デバイス設定 - forkの場合はCPUを強制使用
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# 学習パラメータの設定
lr: float = 2e-3  # 学習率
wd: float = 1e-5  # 重み減衰
betas: Tuple[float, float] = (0.9, 0.999)  # Adamの安定化パラメータ
n_optim: int = 8  # 1バッチ収集あたりの最適化ステップ数（UPD: updates per data）

gamma: float = 0.99  # 割引率
tau: float = 0.02  # ソフトアップデートのパラメータ

total_frames: int = 5_000  # 学習全体のフレーム数（本来は500,000程度）
init_random_frames: int = 100  # 初期ランダムフレーム数（バッファ初期化用）
frames_per_batch: int = 32  # 1バッチあたりのフレーム数
batch_size: int = 32  # バッチサイズ
buffer_size: int = min(total_frames, 100000)  # リプレイバッファのサイズ
num_workers: int = 2  # 並列環境ワーカー数
num_collectors: int = 2  # データコレクター数

eps_greedy_val: float = 0.1  # εグリーディの初期値
eps_greedy_val_env: float = 0.005  # εグリーディの最終値

mp_context: str = multiprocessing.get_start_method()  # マルチプロセスの起動方法

init_bias: float = 2.0  # ネットワークの最終層のバイアス初期値


def is_notebook() -> bool:
    """
    現在の実行環境がJupyter Notebookかどうかを判定する。
    
    Returns:
        bool: Jupyter Notebookの場合はTrue、それ以外はFalse
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def make_env(
    parallel: bool = False,
    obs_norm_sd: Optional[Dict[str, Any]] = None,
    num_workers: int = 1,
) -> TransformedEnv:
    """環境を作成する関数
    
    CartPole-v1環境をピクセル画像ベースで作成し、必要な前処理を適用する。
    
    Args:
        parallel (bool): 環境を並列に実行するかどうか
        obs_norm_sd (Dict[str, Any] | None): 観測値の正規化設定
        num_workers (int): 環境のワーカー数
    
    Returns:
        TransformedEnv: 前処理が適用された環境
    """
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:
        # 並列環境の場合は環境作成関数を定義
        def maker() -> GymEnv:
            return GymEnv(
                "CartPole-v1",
                from_pixels=True,  # ピクセル画像ベースの環境
                pixels_only=True,  # 画像のみを観測値として使用
                device=device,
            )

        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            # workerが1つだけの時は、サブプロセスを作らない
            serial_for_single=True,
            mp_start_method=mp_context,
        )
    else:
        # 単一環境の場合
        base_env = GymEnv(
            "CartPole-v1",
            from_pixels=True,
            pixels_only=True,
            device=device,
        )
    # 環境に変換を適用
    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),  # 各軌道のステップ数をカウント
            ToTensorImage(),  # 画像をPyTorchのテンソルに変換
            GrayScale(),  # グレースケール化
            RewardScaling(loc=0.0, scale=1.0),  # 報酬のスケーリング
            Resize((84, 84)),  # 画像サイズのリサイズ
            CatFrames(4, in_keys=["pixels"], dim=-1),  # 連続する4フレームを連結
            ObservationNorm(in_keys=["pixels"], **obs_norm_sd),  # 観測値の正規化
        ),
    )
    return env


def get_norm_stats() -> Dict[str, Any]:
    """正規化の統計量を取得する関数
    
    ObservationNormのinit_statsメソッドによって、[B, C, H, W]の入力を、
    1. バッチ次元で連結 (cat_dim=0)
    2. Cの次元以外をたたみ込んで統計量を算出 (reduce_dim=[-1, -2, -4])
    3. 計算後にH, W軸を保持 (keep_dims=(-1, -2))
    4. [C, 1, 1]の形状を持つlocとscaleテンソルを得る (state_dictメソッド)
    
    Returns:
        Dict[str, Any]: 観測値の正規化設定
    """
    test_env = make_env()
    test_env.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    print("state dict of the observation norm:", obs_norm_sd)
    test_env.close()
    del test_env
    return obs_norm_sd


def make_model(dummy_env: EnvBase) -> Tuple[QValueActor, TensorDictSequential]:
    """DQNモデルを作成する関数
    
    Args:
        dummy_env (EnvBase): モデル構造を決定するためのダミー環境
    
    Returns:
        Tuple[QValueActor, TensorDictSequential]: 
            - actor: 価値推定のためのQネットワーク
            - actor_explore: 探索機能付きのactor
    """
    # CNNのパラメータ
    cnn_kwargs = {
        "num_cells": [32, 64, 64],  # 各層のフィルタ数
        "kernel_sizes": [8, 4, 3],  # カーネルサイズ
        "strides": [2, 2, 1],  # ストライド
        "activation_class": nn.ELU,  # 活性化関数
    }
    # MLPのパラメータ
    mlp_kwargs = {
        "depth": 2,  # 層の深さ
        "num_cells": [
            64,
            64,
        ],  # 各層のユニット数
        "activation_class": nn.ELU,  # 活性化関数
    }
    # Dueling Network構造のDQNを作成
    net = DuelingCnnDQNet(
        dummy_env.action_spec.shape[-1],  # 行動の数
        1,  # 出力の次元数
        cnn_kwargs,
        mlp_kwargs,
    ).to(device)
    # 最終層のバイアスを設定（初期推定値を高めに）
    net.value[-1].bias.data.fill_(init_bias)

    # 行動選択のためのアクター
    actor = QValueActor(net, in_keys=["pixels"], spec=dummy_env.action_spec).to(device)
    # lazy conv/linear 層で構成されているので、フェイクバッチによる初期化が必要
    tensordict = dummy_env.fake_tensordict()
    actor(tensordict)

    # εグリーディ探索のためのモジュール
    exploration_module = EGreedyModule(
        spec=dummy_env.action_spec,
        annealing_num_steps=total_frames,  # 全フレームで徐々に減衰
        eps_init=eps_greedy_val,  # 初期ε値
        eps_end=eps_greedy_val_env,  # 最終ε値
    )
    # アクターと探索モジュールを結合
    actor_explore = TensorDictSequential(actor, exploration_module)

    return actor, actor_explore


# リプレイバッファの一時ディレクトリ
buffer_scratch_dir = tempfile.TemporaryDirectory().name


def get_replay_buffer(
    buffer_size: int, n_optim: int, batch_size: int, device: torch.device
) -> TensorDictReplayBuffer:
    """リプレイバッファを作成する関数
    
    Args:
        buffer_size (int): バッファのサイズ
        n_optim (int): 最適化のステップ数
        batch_size (int): バッチサイズ
        device (torch.device): テンソルを配置するデバイス
    
    Returns:
        TensorDictReplayBuffer: メモリマップベースのリプレイバッファ
    """
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(buffer_size, scratch_dir=buffer_scratch_dir),
        prefetch=n_optim,  # 事前取得するバッチ数
        transform=lambda td: td.to(device),  # データをGPUに転送
    )
    return replay_buffer


def get_collector(
    stats: Dict[str, Any],
    num_collectors: int,
    actor_explore: TensorDictSequential,
    frames_per_batch: int,
    total_frames: int,
    device: torch.device,
) -> Union[SyncDataCollector, MultiaSyncDataCollector]:
    """データコレクターを作成する関数
    
    環境とのインタラクションを行い、学習データを収集するコレクターを作成する。
    
    Args:
        stats (Dict[str, Any]): 観測値の正規化統計量
        num_collectors (int): コレクターの数
        actor_explore (TensorDictSequential): 探索機能付きアクター
        frames_per_batch (int): バッチあたりのフレーム数
        total_frames (int): 収集する総フレーム数
        device (torch.device): 使用するデバイス
    
    Returns:
        Union[SyncDataCollector, MultiaSyncDataCollector]: データコレクター
    """
    # forkの場合は入れ子のマルチプロセスが使用できないため、同期コレクターを使用
    if is_fork:
        cls = SyncDataCollector
        env_arg = make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
    else:
        cls = MultiaSyncDataCollector
        env_arg = [
            make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
        ] * num_collectors
    data_collector = cls(
        env_arg,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # デフォルトのランダム探索モード
        exploration_type=ExplorationType.RANDOM,
        # すべてのデバイスを同一に設定
        device=device,
        storing_device=device,
        split_trajs=False,  # 軌道の分割を行わない
        postproc=MultiStep(gamma=gamma, n_steps=5),  # n-step学習用の後処理
    )
    return data_collector


def get_loss_module(
    actor: QValueActor, gamma: float
) -> Tuple[DQNLoss, SoftUpdate]:
    """損失関数モジュールとターゲットネットワーク更新機構を作成
    
    Args:
        actor (QValueActor): ポリシーネットワーク
        gamma (float): 割引率
    
    Returns:
        Tuple[DQNLoss, SoftUpdate]: 
            - DQN損失計算モジュール
            - ターゲットネットワーク更新モジュール
    """
    # DQN損失モジュールを作成（ターゲットネットワークを使用）
    loss_module = DQNLoss(actor, delay_value=True)
    # 価値推定器を設定
    loss_module.make_value_estimator(gamma=gamma)
    # ターゲットネットワークの更新機構（ソフトアップデート）
    target_updater = SoftUpdate(loss_module, eps=0.995)
    return loss_module, target_updater


def print_csv_files_in_folder(folder_path: str) -> None:
    """
    フォルダ内のすべてのCSVファイルを見つけて、各ファイルの最初の10行を表示する。
    
    Args:
        folder_path (str): フォルダの相対パス
    """
    csv_files = []
    output_str = ""
    # フォルダ内のすべてのCSVファイルを検索
    for dirpath, _, filenames in os.walk(folder_path):
        for file in filenames:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, file))
    # 各CSVファイルの最初の10行を読み込んで表示
    for csv_file in csv_files:
        output_str += f"File: {csv_file}\n"
        with open(csv_file, "r") as f:
            for i, line in enumerate(f):
                if i == 10:
                    break
                output_str += line.strip() + "\n"
        output_str += "\n"
    print(output_str)


def main() -> None:
    """メイン処理関数
    
    DQN学習の一連の流れを実行する。
    """
    # 観測値の正規化の統計量を計算
    stats = get_norm_stats()
    # テスト環境を作成
    test_env = make_env(parallel=False, obs_norm_sd=stats)
    # モデルを取得
    actor, actor_explore = make_model(test_env)
    # 損失モジュールとターゲットネットワーク更新機構を取得
    loss_module, target_net_updater = get_loss_module(actor, gamma)

    # データコレクターを作成
    collector = get_collector(
        stats=stats,
        num_collectors=num_collectors,
        actor_explore=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )
    # 最適化器を作成
    optimizer = torch.optim.Adam(
        loss_module.parameters(), lr=lr, weight_decay=wd, betas=betas
    )
    # 実験名をUUIDで生成
    exp_name = f"dqn_exp_{uuid.uuid1()}"
    # ログディレクトリを一時フォルダに設定
    tmpdir = tempfile.TemporaryDirectory()
    logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir.name)
    warnings.warn(f"log dir: {logger.experiment.log_dir}")

    # ログの間隔を設定
    log_interval = 500

    # トレーナーを作成
    trainer = Trainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=1,  # フレームスキップなし
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        optim_steps_per_batch=n_optim,  # バッチごとの最適化ステップ数
        log_interval=log_interval,  # ログ取得間隔
    )

    # リプレイバッファを登録
    buffer_hook = ReplayBufferTrainer(
        get_replay_buffer(buffer_size, n_optim, batch_size=batch_size, device=device),
        flatten_tensordicts=True,  # テンソル辞書をフラット化
    )
    buffer_hook.register(trainer)
    
    # 重み更新フックを登録
    weight_updater = UpdateWeights(collector, update_weights_interval=1)
    weight_updater.register(trainer)
    
    # 検証報酬ロガーを登録
    recorder = LogValidationReward(
        record_interval=100,  # 100最適化ステップごとにログ
        record_frames=1000,  # 記録の最大フレーム数
        frame_skip=1,
        policy_exploration=actor_explore,
        environment=test_env,
        exploration_type=ExplorationType.DETERMINISTIC,  # 確定的方策で評価
        log_keys=[("next", "reward")],  # ログに記録するキー
        out_keys={("next", "reward"): "rewards"},  # 出力キーのマッピング
        log_pbar=True,  # プログレスバーにログを表示
    )
    recorder.register(trainer)

    # バッチ収集後にεグリーディのステップを進める操作を登録
    trainer.register_op("post_steps", actor_explore[1].step, frames=frames_per_batch)

    # 最適化後にターゲットネットワークを更新する操作を登録
    trainer.register_op("post_optim", target_net_updater.step)

    # 報酬ロガーを登録
    log_reward = LogScalar(log_pbar=True)
    log_reward.register(trainer)

    # 学習を開始
    trainer.train()
    # 学習結果のCSVファイルを表示
    print_csv_files_in_folder(logger.experiment.log_dir)

    # トレーナーをシャットダウン
    trainer.shutdown()
    del trainer


if __name__ == "__main__":
    main()
