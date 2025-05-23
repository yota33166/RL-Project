import uuid
from dataclasses import dataclass, field
from pathlib import Path
from hydra.core.config_store import ConfigStore


@dataclass
class EnvConfig:
    """Configuration class for the environment."""

    env_name: str = "InvertedDoublePendulum-v4"
    num_workers: int = 4
    reward_scale: float = 0.01
    max_episode_steps: int = 1000
    frame_skip: int = 1
    seed: int = 42


@dataclass
class CollectorConfig:
    """Configuration class for the collector."""

    env_per_collector: int = 4
    frames_per_batch: int = 1024
    total_frames: int = 1_024_000


@dataclass
class OptimConfig:
    """Configuration class for the optimizer."""

    num_cells: int = 256
    lr: float = 3e-4
    wd: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class LossConfig:
    """Configuration class for the loss function."""

    n_optim: int = 5
    sub_batch_size: int = 64
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95
    entropy_eps: float = 1e-4
    critic_coef: float = 1.0


@dataclass
class LogConfig:
    """Configuration class for logging."""

    tb_log_interval: int = 1
    val_record_interval: int = 10
    # exp_name は Hydra ジョブ名をそのまま使う
    exp_name: str = field(default_factory=lambda: "ppo_" + str(uuid.uuid4())[:8])
    # ログ先は Hydra の run.dir を直接参照
    log_dir: Path = Path("logs")
    model_dir: Path = Path("models")


@dataclass
class Config:
    """Configuration class for the agent."""

    load_model: bool = True
    load_model_path: Path = Path("best_model_93.pt")
    env: EnvConfig = field(default_factory=EnvConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    log: LogConfig = field(default_factory=LogConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
