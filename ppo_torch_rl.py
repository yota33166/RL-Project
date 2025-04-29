import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing


from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.trainers import Trainer
from tqdm import tqdm
from tensordict import TensorDict

""" Define Hyperparameters """
is_fork = multiprocessing.get_start_method() == "fork"
device = torch.device("cuda:0" if torch.cuda.is_available() and not is_fork
                      else "cpu")

num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

""" Data collection parameters """
frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 1_000_000

""" PPO parameters """
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

def make_env():
    """ Define the environment """
    base_env = GymEnv("InvertedDoublePendulum-v4", device=device, render_mode="human")

    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    return env

env = make_env()

print("-" * 50)
print("normalization constant shape:", env.transform[0].loc.shape)
print("-" * 50)
print("observation_spec:", env.observation_spec)
print("-" * 50)
print("reward_spec:", env.reward_spec)
print("-" * 50)
print("input_spec:", env.input_spec)
print("-" * 50)
print("action_spec (as defined by input_spec):", env.action_spec)
print("-" * 50)

check_env_specs(env)
print("-" * 50)
rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)
print("-" * 50)

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
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

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))


""" Data collector"""
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)


""" Replay buffer """
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

""" Loss function """
advantage_module = GAE(
    gamma=gamma, 
    lmbda=lmbda, 
    value_network=value_module, 
    average_gae=True, 
    device=device,
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)


""" Training loop """


def train():
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""
    current_frame = 0
    best_eval_reward = -float("inf")
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        num_frames = tensordict_data.numel()  # tensordict_data に含まれるタイムステップ数
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(num_frames)
        current_frame += num_frames
        logs["frames"].append(current_frame)
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                eval_sum = eval_rollout["next", "reward"].sum().item()
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
            if eval_sum > best_eval_reward:
                best_eval_reward = eval_sum
                # 例: モデル、オプティマイザ、スケジューラをまとめて保存
                torch.save({
                    "policy_state_dict": policy_module.state_dict(),
                    "value_state_dict":  value_module.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "obsnorm_state_dict":  env.transform[0].state_dict(),
                    "frame": current_frame,            # 任意で学習ステップ数など
                }, "model_data.pth")
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()


train()

# モデルデータ読み込み
ckpt = torch.load("model_data.pth", map_location=device)
policy_module.load_state_dict(ckpt["policy_state_dict"])
value_module.load_state_dict(ckpt["value_state_dict"])
env.transform[0].load_state_dict(ckpt["obsnorm_state_dict"])
# optim.load_state_dict(ckpt["optimizer_state_dict"])
# scheduler.load_state_dict(ckpt["scheduler_state_dict"])
current_frame = ckpt.get("frame", 0)  # 復元されたフレーム数

# 評価モードに切り替え
policy_module.eval()
value_module.eval()
env.transform[0].eval()

# GymEnv は render() を underlying Gym 環境にフォワードする
# （TorchRL では未定義メソッドは base_env._env に委譲される）
env_td = env.reset()  # TensorDict を返す
done = False

with set_exploration_type(ExplorationType.DETERMINISTIC):
    while not done:
        action = policy_module(env_td)
        # ステップ実行
        next_env_td = env.step(action)
        env.render()
        env_td = next_env_td
        done = bool(next_env_td["next", "done"].item())
