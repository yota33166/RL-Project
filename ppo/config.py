import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvConfig:
    """Configuration class for the environment."""

    env_name: str = "InvertedDoublePendulum-v4"
    num_workers: int = 4
    reward_scale: float = 0.01
    max_episode_steps: int = 1000
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

    log_interval: int = 1
    exp_name: str = f"ppo_{uuid.uuid4().hex[:6]}"
    log_dir: Path = Path("logs") / exp_name
    model_dir: Path = log_dir / "models"


@dataclass
class Config:
    """Configuration class for the agent."""

    load_model: bool = False
    env: EnvConfig = EnvConfig()
    collector: CollectorConfig = CollectorConfig()
    optim: OptimConfig = OptimConfig()
    loss: LossConfig = LossConfig()
    log: LogConfig = LogConfig()
