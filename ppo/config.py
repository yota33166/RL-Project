import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class EnvConfig:
    """環境の設定クラス
    強化学習環境の設定を管理します。
    Args:
        env_name (str): 環境の名前。デフォルトは "InvertedDoublePendulum-v4"。
        num_workers (int): 環境を実行するワーカーの数。デフォルトは 4。
        reward_scale (float): 報酬のスケーリング倍率。デフォルトは 0.01。
        max_episode_steps (int): エピソードの最大ステップ数。デフォルトは 1000。
        frame_skip (int): フレームスキップの間隔。デフォルトは 1。
        seed (int): 環境のシード値。デフォルトは 42。
    """

    env_name: str = "Ant-v4"  # 環境名を変更
    num_workers: int = 16
    reward_scale: float = 0.01
    max_episode_steps: int = 1000
    frame_skip: int = 1
    seed: int = 42


@dataclass
class CollectorConfig:
    """コレクターの設定クラス
    強化学習のデータ収集に関する設定を管理します。
    Args:
        env_per_collector (int): 1つのコレクターが管理する環境の数。デフォルトは 4。
        frames_per_batch (int): 1バッチあたりのフレーム数。デフォルトは 1024。
        total_frames (int): 終了までに収集する総フレーム数。デフォルトは 1,024,000。
    """

    env_per_collector: int = 4
    frames_per_batch: int = 1024
    total_frames: int = 2_048_000


@dataclass
class OptimConfig:
    """最適化の設定クラス
    強化学習の最適化に関する設定を管理します。
    Args:
        num_cells (int): ネットワークの一層あたりのセル数。デフォルトは 256。
        lr (float): 学習率。デフォルトは 3e-4。
        wd (float): 重み減衰。デフォルトは 0.0。
        max_grad_norm (float): 勾配クリッピングの最大ノルム。デフォルトは 1.0。
    """

    num_cells: int = 256
    lr: float = 3e-4
    wd: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class LossConfig:
    """損失関数の設定クラス
    強化学習の損失関数に関する設定を管理します。
    Args:
        n_optim (int): 1バッチあたりの最適化ステップの数。デフォルトは 5。
        sub_batch_size (int): サブバッチのサイズ。デフォルトは 64。
        clip_epsilon (float): PPOのクリッピングイプシロン。デフォルトは 0.2。
        gamma (float): 割引率。デフォルトは 0.99。
        lmbda (float): GAEのラムダ値。デフォルトは 0.95。
        entropy_eps (float): エントロピーのイプシロン。デフォルトは 1e-4。
        critic_coef (float): クリティック損失の係数。デフォルトは 1.0。
    """

    n_optim: int = 5
    sub_batch_size: int = 64
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95
    entropy_eps: float = 1e-4
    critic_coef: float = 1.0


@dataclass
class LogConfig:
    """ロギングの設定クラス
    強化学習のロギングに関する設定を管理します。
    Args:
        tb_log_interval (int): TensorBoardのログ間隔。デフォルトは 1。
        val_record_interval (int): 評価結果の記録間隔。デフォルトは 10。
        exp_name (str): 実験名。デフォルトは "ppo_" にランダムなUUIDを付加したもの。
        log_dir (Path): ログディレクトリのパス。デフォルトは "logs"。
        model_dir (Path): モデルディレクトリのパス。デフォルトは "models"。
    """

    tb_log_interval: int = 1
    val_record_interval: int = 10
    # exp_name は Hydra ジョブ名をそのまま使う
    exp_name: str = field(default_factory=lambda: "ppo_" + str(uuid.uuid4())[:8])
    # ログ先は Hydra の run.dir を直接参照
    log_dir: Path = Path("logs")
    model_dir: Path = Path("models")


@dataclass
class WandbConfig:
    """Weights & Biases の設定"""

    enable: bool = True  # 有効化フラグ
    project: str = "ppo_project"  # W&B プロジェクト名
    offline: bool = False  # オフラインモード
    save_dir: Path = field(default_factory=lambda: Path("wandb"))  # ログ保存先
    id: Optional[str] = None  # 再開用 ID
    video_fps: int = 32


@dataclass
class Config:
    """設定クラスの全体をまとめるクラス
    強化学習の全体的な設定を管理します。
    Args:
        load_model (bool): モデルをロードするかどうか。デフォルトは True。
        load_model_path (Path): ロードするモデルのパス。
        env (EnvConfig): 環境の設定。
        collector (CollectorConfig): コレクターの設定。
        optim (OptimConfig): 最適化の設定。
        loss (LossConfig): 損失関数の設定。
        log (LogConfig): ロギングの設定。
    """

    load_model: bool = False
    load_model_path: Path = Path("best_model_48.pt")
    env: EnvConfig = field(default_factory=EnvConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    log: LogConfig = field(default_factory=LogConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


# HydraのConfigStore に設定クラスを登録
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
