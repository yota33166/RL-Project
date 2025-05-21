from dataclasses import dataclass, asdict
import yaml
from pathlib import Path


@dataclass
class Config:
    """Configuration class for the agent."""

    memo: str = "frames_per_batchを1024に設定"
    SEED = 42
    num_workers: int = 4
    num_collectors: int = 4
    env_name: str = "InvertedDoublePendulum-v4"
    num_cells: int = 256
    lr: float = 3e-4
    wd: float = 0.0
    max_grad_norm: float = 1.0
    reward_scale: float = 0.01

    sub_batch_size: int = 64
    frames_per_batch: int = 1024
    total_frames: int = 1_024_000
    buffer_size: int = min(409_600, total_frames)

    n_optim: int = 5
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95
    alpha_init: float = 0.2

    def save(self):
        """設定を YAML ファイルに保存"""
        with open("RLconfig.yaml", "w") as f:
            yaml.safe_dump(asdict(self), f)

    @classmethod
    def load(cls, path: Path):
        """既存設定ファイルから読み込み、新しいインスタンスを返す"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)