#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")


class BaseConfig:
    """環境やエージェントの設定を管理するクラス

    Attributes:
        env_name (str): 環境名
        num_envs (int): 環境の数
        seed (str): 環境のシード値
        max_epochs (int): エポック数
        eval_interval (int): 評価間隔
        demo_interval (int): デモ間隔
        MODE_DIR (Path): デモ保存先ディレクトリ
        LOG_DIR (Path): ログ保存先ディレクトリ
    """

    # 環境設定
    env_name = "InvertedPendulum-v4"
    num_envs = 8
    seed = 42

    # 訓練設定
    max_epochs = 1000
    eval_interval = 10
    demo_interval = 10

    # ファイル保存場所
    MODEL_DIR = Path("models")
    LOG_DIR = Path("logs")

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Invalid config key: {key}")

        # 保存用ディレクトリの作成
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)

    @property
    def model_save_path(self) -> Path:
        """モデル保存パスを生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.MODEL_DIR) / f"model_{self.env_name}_{timestamp}.pt"

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換（JSONシリアライズ可能な値のみ）"""
        result = {}
        for key, value in self.__dict__.items():
            if (
                isinstance(value, (int, float, str, bool, list, dict, tuple))
                or value is None
            ):
                result[key] = value
        return result

    @classmethod
    def from_json(cls, json_path: str) -> "BaseConfig":
        """JSONファイルから設定を読み込んでインスタンスを生成

        Args:
            json_path (str): 読み込むJSONファイルのパス

        Returns:
            BaseConfig: 生成したインスタンス
        """
        with open(json_path, "r") as f:
            config_data = json.load(f)
        return cls(**config_data)

    def save_json(self, json_path: str) -> None:
        """設定をJSONファイルに保存"""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


class RLLogger:
    """ログの管理クラス

    Attributes:
        config (BaseConfig): 設定オブジェクト
        log_dir (str): ログ保存先ディレクトリ
        logger (Logger): メインロガー
        timimg_logger (Logger): 時間計測ロガー
        writer (SummaryWriter): TensorBoard用のライター
        start_times (dict[str, float]): タスク名とその計測開始時刻を格納した辞書
        durations (dict[str, float]): タスク名とその全エポックの累積時間を格納した辞書
    """

    def __init__(self, log_dir: str, config: BaseConfig):
        self.config = config

        # ログ保存先ディレクトリの作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = config.LOG_DIR / f"{config.env_name}_{timestamp}"
        self.log_dir.mkdir(exist_ok=True)

        # メインロガーの設定
        self.logger = logging.getLogger("RLLogger")
        self.logger.setLevel(logging.INFO)

        # ハンドラの初期化
        if not getattr(self.logger, "_initialized", False):
            self.logger.handlers.clear()
            self.logger._initialized = True

        # コンソールハンドラ
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self.logger.addHandler(console_handler)

        # ファイルハンドラ
        file_handler = logging.FileHandler(self.log_dir / "training.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self.logger.addHandler(file_handler)

        # タイミングロガーの設定
        self.timing_logger = logging.getLogger("TimingLogger")
        self.timing_logger.setLevel(logging.INFO)

        if not getattr(self.timing_logger, "_initialized", False):
            self.timing_logger.handlers.clear()
            self.timing_logger._initialized = True


        timing_handler = logging.FileHandler(self.log_dir / "timing.log")
        timing_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self.timing_logger.addHandler(timing_handler)

        # TensorBoardライターの設定
        self.writer = SummaryWriter(log_dir=self.log_dir / "tensorboard")

        # 時間計測用の変数
        self.start_times = {}
        self.durations = {}

        self.log_start_time("total_training")
        self.log_config()

    def log_config(self, config: Optional[BaseConfig] = None) -> None:
        """設定をログとTensorBoardに記録"""
        if config is None:
            config = self.config

        self.logger.info("-" * 50)
        self.logger.info(
            f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.logger.info(f"Environment: {config.env_name}")

        # 設定値をログに出力
        for key, value in config.to_dict().items():
            self.logger.info(f"{key}: {value}")
            self.writer.add_text(f"Config/{key}", str(value), 0)
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Config/{key}", value, 0)

        self.logger.info("-" * 50)

    def log_start_time(self, task_name: str) -> None:
        """計測開始時刻を記録"""
        self.start_times[task_name] = time.time()

    def log_end_time(self, task_name: str, epoch: Optional[int] = None) -> None:
        """計測終了時刻を記録し、経過時間を返す"""
        if task_name not in self.start_times:
            self.timing_logger.warning(f"Start time for {task_name} not found.")
            return 0.0

        elapsed_time = time.time() - self.start_times[task_name]
        if epoch is not None:
            # エポック単位でログとTensorBoardに記録
            self.timing_logger.info(
                f"Epoch {epoch}: {task_name} took {elapsed_time:.4f} seconds"
            )
            self.writer.add_scalar(f"Timing/{task_name}", elapsed_time, epoch)

            # 累積時間を記録
            if task_name not in self.durations:
                self.durations[task_name] = 0.0
            self.durations[task_name] += elapsed_time
        else:
            # エポック数が指定されていない場合は、全体の経過時間を記録
            self.timing_logger.info(f"{task_name} took {elapsed_time:.4f} seconds")

        return elapsed_time

    def log_summary(self) -> None:
        """ "全体の経過時間をログとTensorBoardに記録"""
        total_time = self.log_end_time("total_training")

        self.logger.info("-" * 50)
        self.timing_logger.info("-" * 50)
        self.logger.info(
            f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.timing_logger.info(
            f"Total training time: {timedelta(seconds=total_time)} seconds"
        )
        self.writer.add_scalar("Timing/Total Training Time", total_time, 0)

        # 各タスクの累積時間をログとTensorBoardに記録
        self.logger.info("\nTask durations:")
        for task_name, duration in self.durations.items():
            percentage = (duration / total_time) * 100
            self.logger.info(
                f"{task_name}: {timedelta(seconds=duration)} seconds \
                ({percentage:.2f}%)"
            )
            self.timing_logger.info(
                f"{task_name}: {timedelta(seconds=duration)} seconds \
                ({percentage:.2f}%)"
            )
            self.writer.add_scalar(f"Timing/{task_name}", duration, 0)

        self.logger.info("-" * 50)
        self.timing_logger.info("-" * 50)

    def info(self, message: str) -> None:
        """情報メッセージをログに記録"""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """デバッグメッセージをログに記録"""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """警告メッセージをログに記録"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """エラーをログに記録"""
        self.logger.error(message)

    def log_metric(self, metrics: Dict[str, float], epoch: int) -> None:
        """メトリックをTensorBoardに記録"""
        # メトリクスをログに出力
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metrics_str}")

        # TensorBoardに記録
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)

    def close(self):
        self.logger.close()
        self.timing_logger.close()
        self.writer.close()
