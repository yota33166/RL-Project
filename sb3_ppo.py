#!/usr/bin/env python3
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main():
    # 並列環境の作成（CPU数-1を利用）
    num_cpu = multiprocessing.cpu_count() - 1
    env = make_vec_env(
        "HalfCheetah-v4",
        n_envs=num_cpu,
    )

    # PPOモデルの定義
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1000,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=1e-4,
        clip_range=0.2,
        tensorboard_log="./tensorboard/",
        device="auto",
    )

    # 学習実行
    model.learn(total_timesteps=1_000_000)

    # モデルと環境の保存・クローズ
    model.save("ppo_halfcheetah")
    env.close()


if __name__ == "__main__":
    main()
