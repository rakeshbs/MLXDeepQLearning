"""
Experiment: Flappy Bird + Double DQN (parallel)

Identical setup to flappy_dqn.py but uses DoubleDQN instead of DQN.
The only difference is that _compute_targets() uses the online net to
select the best next action and the target net to evaluate it, which
reduces Q-value overestimation and typically leads to faster convergence.

Train:  python -m experiments.flappy.double_dqn
Test:   python -m experiments.flappy.double_dqn --test [--best]
"""

import os
import sys

from algorithms.dqn import DQNConfig, MLPQNetwork
from algorithms.double_dqn import DoubleDQN
from envs.flappy_bird import FlappyBirdEnv
from training.parallel_runner import ParallelRunner
from training.runner import RunnerConfig


def gap_reward(env, reward, done):
    """
    Per-step hint: add a small bonus proportional to how close the bird is
    to the center of the upcoming pipe gap.

    proximity is in [0, 1] where 1 = perfectly centered. The 0.1 scale keeps
    the shaping reward small relative to the +10 pipe-clear reward so it guides
    exploration without overshadowing the true game objective.
    """
    if done or not env.pipes:
        return reward
    next_pipe = next(
        (p for p in env.pipes if p["x"] + env.pipe_width > env.bird_x), None
    )
    if next_pipe is None:
        return reward
    gap_center = (next_pipe["top"] + next_pipe["bottom"]) / 2
    proximity  = 1.0 - abs(env.bird_y - gap_center) / env.screen_height
    return reward + 0.1 * proximity


def make_algo():
    """
    Factory function that constructs a fresh DoubleDQN instance.

    Must be a named module-level function (not a lambda) so it can be
    pickled and sent to spawned actor processes by multiprocessing.
    """
    return DoubleDQN(DQNConfig(
        action_dim=2,
        network_factory=lambda: MLPQNetwork(state_dim=5, hidden_dim=128, action_dim=2),
        lr=1e-3,
        gamma=0.99,
    ))


runner = ParallelRunner(
    env_factory=FlappyBirdEnv,
    algo=make_algo(),       # learner's copy of the algorithm
    algo_factory=make_algo, # factory passed to actor processes for their own copies
    config=RunnerConfig(
        buffer_size=100_000,
        batch_size=64,
        train_start=1_000,
        max_episodes=1_000_000,
        epsilon_start=1.0,
        epsilon_end=0.001,
        epsilon_decay=0.999,
        render_every=200,
        ckpt_dir=os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "flappy_double_dqn"),
        log_every=50,
    ),
    num_actors=6,
    updates_per_drain=4,
    weight_sync_freq=100,
    epsilon_base=0.4,
    epsilon_base_decay=0.999,
    epsilon_base_min=0.02,
    reward_shaper=gap_reward,
    per_alpha=0.3,
    per_beta=0.6,
)

if __name__ == "__main__":
    if "--test" in sys.argv:
        runner.test(best="--best" in sys.argv)
    else:
        runner.train()
