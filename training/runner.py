import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

import mlx.core as mx

from algorithms.base import BaseAlgorithm
from algorithms.buffers import ReplayBuffer
from envs.base import BaseEnv
from training.checkpoint import Checkpointer


@dataclass
class RunnerConfig:
    buffer_size: int = 50_000
    batch_size: int = 64
    train_start: int = 1_000  # steps before learning begins (pure exploration)
    max_episodes: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.001
    epsilon_decay: float = 0.999  # multiplied every episode (geometric decay)
    render_every: int = 200  # render one episode every N episodes
    ckpt_dir: str = "checkpoints"
    log_every: int = 50


class Runner:
    """
    Single-process training loop for one algorithm on one environment.

    Knows nothing about game physics or algorithm internals — it only calls
    the BaseEnv and BaseAlgorithm interfaces and manages epsilon, the replay
    buffer, and checkpointing. This separation keeps the runner reusable
    across different games and algorithms.

    reward_shaper: optional callable(env, reward, done) -> reward
        Use this for environment-specific reward shaping (e.g. gap proximity
        for Flappy Bird). Defined in the experiment file, not here, so the
        runner stays generic and testable in isolation.
    """

    def __init__(
        self,
        env_factory: Type[BaseEnv],
        algo: BaseAlgorithm,
        config: RunnerConfig,
        reward_shaper: Optional[Callable] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.env_factory = env_factory
        self.algo = algo
        self.config = config
        self.reward_shaper = reward_shaper
        self.env_kwargs = env_kwargs or {}
        self.buffer = ReplayBuffer(config.buffer_size)
        self.checkpointer = Checkpointer(config.ckpt_dir)

    def train(self):
        self.checkpointer.install_process_logger()
        self._train_impl()

    def _train_impl(self):
        """
        Main training loop. Runs for config.max_episodes episodes.

        On startup, attempts to resume from the latest checkpoint. If one
        exists, epsilon and total_steps are restored so training continues
        as if it never stopped. The environment is re-created each episode so
        resources are cleanly released between episodes.
        """
        cfg = self.config

        # Try to restore previous training state from disk
        meta = self.checkpointer.load(self.algo)
        if meta:
            start_ep = meta["episode"] + 1
            epsilon = meta["epsilon"]
            total_steps = meta["total_steps"]
            best_score = meta["best_score"]
            print(
                f"Resuming from checkpoint — "
                f"episode={start_ep}  epsilon={epsilon:.4f}  "
                f"total_steps={total_steps}  best_score={best_score}"
            )
        else:
            print("No checkpoint found — starting fresh.")
            start_ep = 1
            epsilon = cfg.epsilon_start
            total_steps = 0
            best_score = -1

        for episode in range(start_ep, start_ep + cfg.max_episodes):
            # Render every N-th episode so human oversight is possible without
            # paying the rendering overhead on every episode
            render = episode % cfg.render_every == 0
            env = self.env_factory(render_mode=render, **self.env_kwargs)
            state = env.reset()
            episode_reward = 0.0

            while True:
                # Epsilon-greedy: explore randomly during warm-up or with prob epsilon
                if random.random() < epsilon or total_steps < cfg.train_start:
                    action = random.randint(0, env.action_dim - 1)
                else:
                    action = self.algo.select_action(state)

                next_state, reward, done, info = env.step(action)

                # Apply optional reward shaping (e.g. proximity bonus for Flappy Bird)
                if self.reward_shaper is not None:
                    reward = self.reward_shaper(env, reward, done)

                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                total_steps += 1

                # Only start learning once the buffer has enough diverse experience
                if len(self.buffer) >= cfg.train_start:
                    batch = self.buffer.sample(cfg.batch_size)
                    self.algo.update(batch)

                # Periodically release the MLX memory pool to avoid fragmentation
                if total_steps % 200 == 0:
                    mx.clear_cache()

                if done:
                    break

            env.close()
            mx.clear_cache()  # clean up GPU memory between episodes

            # Geometric epsilon decay: explore less as training matures
            epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

            score = info["score"]
            is_best = score > best_score
            if is_best:
                best_score = score

            self.checkpointer.save(
                self.algo,
                meta={
                    "episode": episode,
                    "epsilon": round(epsilon, 6),
                    "total_steps": total_steps,
                    "best_score": best_score,
                },
                is_best=is_best,
            )

            if episode % cfg.log_every == 0:
                print(
                    f"Episode {episode:5d} | "
                    f"Score: {score:3d} | "
                    f"Best: {best_score:3d} | "
                    f"Reward: {episode_reward:7.1f} | "
                    f"Epsilon: {epsilon:.4f} | "
                    f"Steps: {total_steps}"
                )

        print(f"\nCycle complete. Best score: {best_score}")

    def test(self, best: bool = False):
        self.checkpointer.install_process_logger()
        self._test_impl(best)

    def _test_impl(self, best: bool = False):
        """
        Run a checkpoint greedily with rendering until KeyboardInterrupt.

        Uses the 'best' checkpoint when best=True, otherwise 'latest'.
        No epsilon — the policy is always greedy so performance reflects
        what the agent actually learned.
        """
        if best:
            meta = self.checkpointer.load_best(self.algo)
            tag = "best"
        else:
            meta = self.checkpointer.load(self.algo)
            tag = "latest"

        if meta is None:
            print(f"No {tag} checkpoint found.")
            return

        print(
            f"Loaded {tag} checkpoint — "
            f"trained for {meta['episode']} episodes, "
            f"best score: {meta['best_score']}"
        )

        episode = 0
        try:
            while True:
                env = self.env_factory(render_mode=True, **self.env_kwargs)
                state = env.reset()
                episode += 1

                while True:
                    action = self.algo.select_action(state)
                    state, _, done, info = env.step(action)
                    if done:
                        print(f"Episode {episode} — Score: {info['score']}")
                        break

                env.close()
        except KeyboardInterrupt:
            pass
