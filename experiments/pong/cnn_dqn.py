"""
Experiment: Pong — right paddle trained with Double DQN on pixel observations.

The left paddle is a simple speed-capped ball-tracker (rudimentary AI).
The right paddle is the learning agent.

Obs:  4 stacked 84×84 grayscale frames — shape (4, 84, 84), stored as uint8.
Net:  CNNQNetwork (3 conv layers + MLP head, standard Atari architecture).
Algo: Double DQN (separate selection/evaluation nets to reduce overestimation).

Train:  python -m experiments.pong.cnn_dqn
Test:   python -m experiments.pong.cnn_dqn --test [--best]
"""

import os
import sys

from algorithms.double_dqn import DoubleDQN
from algorithms.dqn import CNNQNetwork, DQNConfig
from envs.pong import PongEnv
from training.parallel_runner import ParallelRunner
from training.runner import RunnerConfig

OBS_SHAPE  = (4, 84, 84)
ACTION_DIM = 3


def make_algo():
    return DoubleDQN(DQNConfig(
        action_dim=ACTION_DIM,
        network_factory=lambda: CNNQNetwork(obs_shape=OBS_SHAPE, action_dim=ACTION_DIM),
        lr=1e-4,
        gamma=0.99,
        target_update_freq=2_500,
    ))


runner = ParallelRunner(
    env_factory=PongEnv,
    algo=make_algo(),
    algo_factory=make_algo,
    config=RunnerConfig(
        buffer_size=100_000,
        batch_size=32,
        train_start=5_000,
        max_episodes=500_000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999,
        render_every=500,
        ckpt_dir=os.path.join(
            os.path.dirname(__file__), "..", "..", "checkpoints", "pong_cnn_dqn"
        ),
        log_every=100,
    ),
    num_actors=6,
    updates_per_drain=4,
    weight_sync_freq=200,
    epsilon_base=0.4,
    epsilon_base_decay=0.9999,
    epsilon_base_min=0.05,
    per_alpha=0.6,
    per_beta=0.4,
    env_kwargs={"obs_type": "pixels", "score_limit": 5, "max_steps": 3_000},
)

if __name__ == "__main__":
    if "--test" in sys.argv:
        runner.test(best="--best" in sys.argv)
    else:
        runner.train()
