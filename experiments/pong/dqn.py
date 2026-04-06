"""
Experiment: Pong — right paddle trained with standard DQN.

The left paddle is a simple speed-capped ball-tracker (rudimentary AI).
The right paddle is the learning agent.

State: 6-dim float vector
    (ball_x, ball_y, ball_vx, ball_vy, right_paddle_cy, left_paddle_cy)
    all normalised to roughly [-1, 1].

Actions: 0=stay, 1=up, 2=down

Reward: +1 per point scored, -1 per point conceded.

Train:  python -m experiments.pong.dqn
Test:   python -m experiments.pong.dqn --test [--best]
"""

import os
import sys

from algorithms.dqn import DQN, DQNConfig, MLPQNetwork
from envs.pong import PongEnv
from training.runner import Runner, RunnerConfig


def make_algo():
    return DQN(DQNConfig(
        action_dim=3,
        network_factory=lambda: MLPQNetwork(state_dim=6, hidden_dim=256, action_dim=3),
        lr=5e-4,
        gamma=0.99,
        target_update_freq=1_000,
    ))


runner = Runner(
    env_factory=PongEnv,
    algo=make_algo(),
    config=RunnerConfig(
        buffer_size=100_000,
        batch_size=128,
        train_start=2_000,
        max_episodes=50_000,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.9995,
        render_every=500,
        ckpt_dir=os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "pong_dqn"),
        log_every=100,
    ),
    env_kwargs={"score_limit": 5, "max_steps": 3_000},
)

if __name__ == "__main__":
    if "--test" in sys.argv:
        runner.test(best="--best" in sys.argv)
    else:
        runner.train()
