#!/usr/bin/env python3
import atexit
import multiprocessing
import os
import signal
import sys
import warnings
from datetime import datetime
from pathlib import Path

import hydra
import torch
from config import Config
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from torch.multiprocessing import freeze_support
from torchrl._utils import _CKPT_BACKEND
from torchrl.envs.utils import ExplorationType
from torchrl.objectives.value import GAE
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.trainers import (
    BatchSubSampler,
    ClearCudaCache,
    LogScalar,
    Trainer,
    UpdateWeights,
)
from utils import (
    LogLearningRate,
    SaveBestValidationReward,
    _signal_handler,
    cleanup_resources,
    get_collector,
    get_loss_module,
    get_norm_stats,
    load_and_demo_model,
    make_env,
    make_ppo_model,
)

warnings.filterwarnings("ignore")


@hydra.main(config_path=None, config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))

    # 並列処理のコンテキストを取得
    mp_context = multiprocessing.get_start_method()
    is_fork = mp_context == "fork"

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not is_fork else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Using multiprocessing context: {mp_context}")

    freeze_support()  # Windows の場合、マルチプロセスのために必要
    num_workers = cfg.env.num_workers
    # num_workers = multiprocessing.cpu_count()  # number of workers for parallel envs
    print(f"Number of workers: {num_workers}")
    num_collectors = (
        cfg.collector.env_per_collector
    )  # number of collectors for parallel envs
    print(f"Number of collectors: {num_collectors}")

    # 学習済みモデルをロードする場合
    if cfg.load_model:
        # ppo_InvertedDoublePendulum-v4_May15_10-29-00\best_model_93.pt
        model_path = Path(get_original_cwd()) / cfg.load_model_path
        load_and_demo_model(cfg, model_path, device, iteration=1)
        cleanup_resources()
        sys.exit(0)

    # トレーナー保存先のディレクトリを作成
    if _CKPT_BACKEND == "torchsnapshot":
        save_trainer_file = (
            cfg.log.model_dir
            / "trainer"
            / f"{datetime.now().strftime('%b%d_%H-%M-%S')}",
        )
    elif _CKPT_BACKEND == "torch":
        save_trainer_file = (
            cfg.log.model_dir
            / "trainer"
            / f"{datetime.now().strftime('%b%d_%H-%M-%S')}.pt"
        )
    else:
        raise ValueError(f"Unknown checkpoint backend: {_CKPT_BACKEND}")

    run_dir = Path(os.getcwd())
    logger = TensorboardLogger(exp_name=cfg.log.exp_name, log_dir=run_dir)
    # モデル保存先が model_dir: "${hydra.run.dir}/models" なので
    (run_dir / "models").mkdir(exist_ok=True)

    # TensorBoardに設定情報を追加
    # for key, value in asdict(cfg).items():
    #     logger.experiment.add_text(key, str(value))

    test_env = make_env(cfg, device=device)
    stats = get_norm_stats(test_env)
    test_env = make_env(cfg, device=device, mp_context=mp_context, obs_norm_sd=stats)
    policy_module, value_module = make_ppo_model(cfg.optim.num_cells, test_env, device)

    print("policy", policy_module(test_env.reset()))
    print("value", value_module(test_env.reset()))

    loss_module = get_loss_module(
        cfg,
        actor_network=policy_module,
        critic_network=value_module,
    )
    collector = get_collector(
        cfg,
        mp_context=mp_context,
        stats=stats,
        policy=policy_module,
        device=device,
    )

    optim = torch.optim.Adam(
        loss_module.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, cfg.collector.total_frames // cfg.collector.frames_per_batch, 0.0
    )

    trainer = Trainer(
        collector=collector,
        total_frames=cfg.collector.total_frames,
        frame_skip=cfg.env.frame_skip,
        loss_module=loss_module,
        optimizer=optim,
        logger=logger,
        optim_steps_per_batch=cfg.loss.n_optim,
        log_interval=cfg.log.tb_log_interval,
        clip_norm=cfg.optim.max_grad_norm,
        seed=cfg.env.seed,
    )

    advantage_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.lmbda,
        value_network=value_module,
        average_gae=True,
        device=device,
    )

    trainer.register_op("batch_process", advantage_module)
    batch_subsampler = BatchSubSampler(
        batch_size=cfg.loss.sub_batch_size,
        sub_traj_len=1,
    )
    batch_subsampler.register(trainer)

    # buffer_hook = ReplayBufferTrainer(
    #     # PPOはオンポリシーアルゴリズムなので、buffer_sizeはframes_per_batchと同じ
    #     get_replay_buffer(
    #         frames_per_batch, n_optim, batch_size=sub_batch_size, device=device
    #     ),
    #     flatten_tensordicts=True,
    # )
    # buffer_hook.register(trainer)

    weight_updater = UpdateWeights(collector=collector, update_weights_interval=1)
    weight_updater.register(trainer)
    recorder = SaveBestValidationReward(
        cfg=cfg,
        value_module=value_module,
        record_interval=cfg.log.val_record_interval,
        record_frames=cfg.env.max_episode_steps,
        frame_skip=cfg.env.frame_skip,
        policy_exploration=policy_module,
        environment=test_env,
        exploration_type=ExplorationType.DETERMINISTIC,
        log_keys=[("next", "reward")],
        out_keys={("next", "reward"): "r_evaluation"},
        log_pbar=True,
    )
    recorder.register(trainer)

    trainer.register_op("post_steps", scheduler.step)

    log_reward = LogScalar(log_pbar=True)
    log_reward.register(trainer)

    log_lr = LogLearningRate(optim, logger, log_interval=cfg.log.tb_log_interval)
    log_lr.register(trainer)

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


if __name__ == "__main__":
    main()
