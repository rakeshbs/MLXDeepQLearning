"""
Experiment: Flappy Bird + Standard DQN (parallel)

Uses 6 parallel actors with a 5-dim state observation (no pixels).
The gap_reward shaper adds a small proximity bonus every step to guide
exploration toward the pipe gap before the agent discovers scoring by itself.

Train:  python -m experiments.flappy.dqn
Test:   python -m experiments.flappy.dqn --test [--best]
"""

import os
import sys

from algorithms.dqn import DQN, DQNConfig, MLPQNetwork
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
    Factory function that constructs a fresh DQN instance.

    Must be a named module-level function (not a lambda) so it can be
    pickled and sent to spawned actor processes by multiprocessing.
    """
    return DQN(DQNConfig(
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
        ckpt_dir=os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "flappy_dqn"),
        log_every=50,
    ),
    num_actors=6,
    updates_per_drain=4,
    weight_sync_freq=100,
    epsilon_base=0.4,         # starting base for the epsilon spread; decays over training
    epsilon_base_decay=0.999, # ×0.999 per weight sync → ~0.13 after 2000 syncs
    epsilon_base_min=0.02,    # floor: keep some diversity forever so actors don't all go greedy
    reward_shaper=gap_reward,
    per_alpha=0.3,   # low alpha = mild prioritisation (closer to uniform)
    per_beta=0.6,    # IS weight correction strength
)

if __name__ == "__main__":
    if "--test" in sys.argv:
        runner.test(best="--best" in sys.argv)
    else:
        runner.train()
