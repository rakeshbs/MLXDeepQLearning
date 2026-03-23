"""
Experiment: Flappy Bird + CNN DQN (parallel, pixel observations)

Obs: 4 stacked 84×84 grayscale frames — shape (4, 84, 84)
Net: CNNQNetwork (3 conv layers + MLP head, standard Atari architecture)

Compared to the MLP state-vector experiments:
  - Smaller batch size (32) because pixel batches use much more GPU memory.
  - Higher train_start (5000) to ensure the PER buffer has diverse visual
    experiences before gradient updates begin.
  - 8 actors — beyond ~8 actors the shared Metal GPU becomes the bottleneck
    (actors compete with the learner).
  - Slower epsilon_base_decay (0.9995 vs 0.999) because pixel-based policies
    take longer to converge and need sustained exploration.
  - Gap proximity reward shaping: each step adds a bonus for being vertically
    aligned with the next pipe gap center. This makes rewards dense enough for
    the CNN to get a useful gradient signal even in short early episodes.
  - Tuned hyperparams: per_alpha=0.6, per_beta=0.4, target_update=2500.

Train:  python -m experiments.flappy.cnn_dqn
Test:   python -m experiments.flappy.cnn_dqn --test [--best]
"""

import os
import sys

from algorithms.double_dqn import DoubleDQN
from algorithms.dqn import DQNConfig, CNNQNetwork
from envs.flappy_bird import FlappyBirdEnv
from training.parallel_runner import ParallelRunner
from training.runner import RunnerConfig

OBS_SHAPE  = (4, 84, 84)  # (frames_stacked, height, width)
ACTION_DIM = 2


def gap_proximity_shaper(env, reward, done):
    """
    Add a small per-step bonus for being vertically aligned with the next pipe
    gap center, then clip the total reward to [-1, +1].

    The bonus is capped at 0.02 — just enough to provide a weak alignment signal
    without creating a local optimum where surviving near the gap is more
    rewarding than actually passing it. Reward clipping bounds Q-values and
    prevents the proximity bonus from dominating the pipe-pass reward (+10→+1).
    """
    next_pipe = None
    for pipe in env.pipes:
        if pipe["x"] + env.pipe_width > env.bird_x:
            next_pipe = pipe
            break

    if next_pipe is not None:
        gap_center  = (next_pipe["top"] + next_pipe["bottom"]) / 2
        bird_center = env.bird_y + env.bird_height / 2
        dist        = abs(bird_center - gap_center)
        # proximity_bonus in [0, 0.02]: 0.02 when perfectly centred, 0 at screen edge
        proximity_bonus = 0.02 * max(0.0, 1.0 - dist / env.screen_height)
        reward += proximity_bonus

    return float(max(-1.0, min(1.0, reward)))


def make_algo():
    """
    Factory function that constructs a fresh CNN DQN instance.

    Must be a named module-level function (not a lambda) so it can be
    pickled and sent to spawned actor processes by multiprocessing.
    lr=1e-4 (vs 1e-3 for MLP) is standard for CNN-based DQN — the larger
    gradient magnitudes from conv layers benefit from a smaller step size.
    """
    return DoubleDQN(DQNConfig(
        action_dim=ACTION_DIM,
        network_factory=lambda: CNNQNetwork(obs_shape=OBS_SHAPE, action_dim=ACTION_DIM),
        lr=1e-4,
        gamma=0.99,
        target_update_freq=2_500,
    ))


runner = ParallelRunner(
    env_factory=FlappyBirdEnv,
    algo=make_algo(),       # learner's copy of the algorithm
    algo_factory=make_algo, # factory passed to actor processes for their own copies
    config=RunnerConfig(
        buffer_size=100_000,  # doubled from 50k for more diversity
        batch_size=32,
        train_start=5_000,
        max_episodes=1_000_000,
        epsilon_start=1.0,
        epsilon_end=0.001,
        epsilon_decay=0.999,
        render_every=200,
        ckpt_dir=os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "flappy_cnn_dqn"),
        log_every=50,
    ),
    reward_shaper=gap_proximity_shaper,
    num_actors=8,             # 8 actors without starving the learner GPU
    epsilon_alpha=4.0,        # 7.0 assumes 360 actors; with 8 actors
                              # it puts 5/8 actors at ε<0.026 before the policy learns anything
    updates_per_drain=4,      # more actors → more data → more updates per cycle
    weight_sync_freq=200,
    epsilon_base=0.4,
    epsilon_base_decay=0.9999, # slowly decay toward greedy as policy matures
    epsilon_base_min=0.05,    # floor — always keep some exploration
    per_alpha=0.6,
    per_beta=0.4,
    env_kwargs={"obs_type": "pixels"},  # tells FlappyBirdEnv to return pixel observations
)

if __name__ == "__main__":
    if "--test" in sys.argv:
        runner.test(best="--best" in sys.argv)
    else:
        runner.train()
