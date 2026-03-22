"""
Experiment: Breakout + CNN DQN (parallel, pixel observations)

Obs: 4 stacked 84x84 grayscale frames — shape (4, 84, 84)
Net: CNNQNetwork (3 conv layers + MLP head, DeepMind Atari architecture)
Algo: Double DQN

The environment follows the Atari Breakout scoring and match structure:
  - 4 actions: NOOP, FIRE, RIGHT, LEFT
  - 5 lives
  - 2 walls total per game
  - brick rewards by row: 7, 7, 4, 4, 1, 1

Train:  python -m experiments.breakout_cnn_dqn
Test:   python -m experiments.breakout_cnn_dqn --test [--best] [--full-game] [--episodes=N] [--no-render]
"""

import os
import sys

import numpy as np

from algorithms.double_dqn import DoubleDQN
from algorithms.dqn import CNNQNetwork, DQNConfig
from envs.breakout import BreakoutEnv
from training.parallel_runner import ParallelRunner
from training.runner import RunnerConfig

OBS_SHAPE = (4, 84, 84)
ACTION_DIM = 4


def make_algo():
    return DoubleDQN(
        DQNConfig(
            action_dim=ACTION_DIM,
            network_factory=lambda: CNNQNetwork(
                obs_shape=OBS_SHAPE, action_dim=ACTION_DIM
            ),
            lr=1e-4,
            gamma=0.99,
            target_update_freq=10_000,
        )
    )


def reward_shaper(env, reward, done):
    return float(np.clip(reward, -1.0, 1.0))


runner = ParallelRunner(
    env_factory=BreakoutEnv,
    algo=make_algo(),
    algo_factory=make_algo,
    config=RunnerConfig(
        buffer_size=1_000_000,
        batch_size=64,
        train_start=50_000,
        max_episodes=1_000_000,
        render_every=200,
        ckpt_dir=os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "breakout_cnn_dqn"
        ),
        log_every=50,
    ),
    reward_shaper=reward_shaper,
    num_actors=8,
    updates_per_drain=1,
    weight_sync_freq=200,
    epsilon_alpha=4.0,
    epsilon_base=0.4,
    epsilon_base_decay=0.9997,
    epsilon_base_min=0.2,
    per_alpha=0.6,
    per_beta=0.4,
    lr_decay=1.0,
    lr_min=1e-4,
    per_beta_increment=0.0001,
    actor_random_warmup_steps=6_250,
    env_kwargs={"obs_type": "pixels", "frame_skip": 4},
)

if __name__ == "__main__":
    if "--test" in sys.argv:
        full_game = "--full-game" in sys.argv
        render = "--no-render" not in sys.argv
        num_episodes = 0
        for arg in sys.argv:
            if arg.startswith("--episodes="):
                num_episodes = int(arg.split("=")[1])
        runner.test(
            best="--best" in sys.argv,
            best_score="--best-score" in sys.argv,
            env_kwargs_override={"terminal_on_life_loss": False} if full_game else None,
            num_episodes=num_episodes,
            render=render,
        )
    else:
        runner.train()
