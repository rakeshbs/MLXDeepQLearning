import multiprocessing as mp
import os
import random
import resource
import threading
import time
from typing import Any, Callable, Dict, Optional

import mlx.core as mx
import numpy as np

from algorithms.buffers import PrioritizedReplayBuffer, ReplayBuffer
from training.checkpoint import Checkpointer, install_process_logger
from training.runner import RunnerConfig

_BOOT_LOG_PATH = os.environ.get("MLX_RL_LOG_PATH")
if _BOOT_LOG_PATH:
    install_process_logger(_BOOT_LOG_PATH)


def _ape_x_epsilons(num_actors: int, base: float = 0.4, alpha: float = 7.0) -> list:
    """
    Ape-X epsilon schedule: actor i gets ε_i = base ^ (1 + α * i / (N - 1)).

    This formula spreads actors across a wide exploration range in a single
    parameter (base). Actor 0 is the most exploratory (base^1 = base) and
    actor N-1 is nearly greedy (base^(1+alpha) → 0 for large alpha). The
    diversity of experience collected across actors improves sample efficiency
    compared to all actors sharing the same epsilon. When epsilon_base decays
    over training, the entire spread shifts toward more greedy behaviour.
    """
    if num_actors == 1:
        return [base]
    return [base ** (1.0 + alpha * i / (num_actors - 1)) for i in range(num_actors)]


def _actor_fn(
    actor_id: int,
    env_factory,
    algo_factory,
    reward_shaper,
    weight_queue,
    transition_queue,
    epsilon: float,
    random_warmup_steps: int,
    env_kwargs: dict,
    initial_weights: dict,
    ckpt_dir: str,
    skip_random_warmup: bool = False,
):
    """
    Runs in a child process. Each actor has its own independent MLX Metal context.

    The actor runs an infinite episode loop, collecting (s, a, r, s', done)
    transitions and pushing them into the shared transition_queue for the
    learner process to consume. If the queue is full, transitions are dropped
    rather than blocking — actor throughput is more important than perfect
    capture rate.

    Weight updates arrive as (weights_dict, new_epsilon) tuples on weight_queue.
    Draining the queue before each episode ensures the actor uses the latest
    learner weights without blocking the episode loop mid-step.

    epsilon starts at the Ape-X assigned value and is updated by the learner
    via weight_queue messages, which are (weights_dict, new_epsilon) tuples.
    """
    # Import mlx inside the child process — each spawned process gets a fresh
    # Metal GPU context, which is why 'spawn' (not 'fork') is required.
    import mlx.core as mx

    Checkpointer(ckpt_dir).install_process_logger()

    algo = algo_factory()
    algo.set_weights(initial_weights)
    env = env_factory(**env_kwargs)
    total_steps = random_warmup_steps if skip_random_warmup else 0

    while True:
        # Drain weight_queue before each episode — only keep the most recent
        # message so stale weight broadcasts don't accumulate. Using get_nowait
        # means this never blocks the actor.
        latest_weights = None
        while not weight_queue.empty():
            try:
                msg = weight_queue.get_nowait()
                if isinstance(msg, tuple):
                    latest_weights, epsilon = msg  # unpack (weights, new_epsilon)
                else:
                    latest_weights = msg  # legacy scalar message (backward compat)
            except Exception:
                break
        if latest_weights is not None:
            algo.set_weights(latest_weights)

        state = env.reset()
        while True:
            # Epsilon-greedy: random action during actor warm-up or with prob epsilon
            if random.random() < epsilon or total_steps < random_warmup_steps:
                action = random.randint(0, env.action_dim - 1)
            else:
                action = algo.select_action(state)

            next_state, reward, done, info = env.step(action)

            if reward_shaper is not None:
                reward = reward_shaper(env, reward, done)

            # Only send the score on terminal transitions — the learner uses it
            # to track episode boundaries and update the best-score tracker.
            score = info.get("score") if done else None
            transition = (state, action, float(reward), next_state, bool(done), score)
            if done:
                try:
                    transition_queue.put(transition, timeout=5.0)
                except Exception:
                    pass  # learner appears dead; keep actor alive
            else:
                try:
                    transition_queue.put_nowait(transition)
                except Exception:
                    pass  # queue full — drop transition, never block
            state = next_state
            total_steps += 1

            if done:
                break

        # Release GPU memory between episodes; each episode's computation graph
        # has already been materialised so clearing the cache is safe here.
        mx.clear_cache()


class ParallelRunner:
    """
    Ape-X style parallel training loop.

    Architecture:
      - N actor processes run inference-only loops, each with its own MLX
        Metal GPU context (spawned via 'spawn', not 'fork', to avoid shared
        Metal state). Actors push transitions into a shared multiprocessing Queue.
      - One learner (this process) owns the PrioritizedReplayBuffer, optimizer,
        and target network. It drains the queue in batches, runs gradient updates,
        and periodically broadcasts updated weights back to actors.

    Communication:
      - transition_queue: actors → learner (s, a, r, s', done, score)
      - weight_queues[i]: learner → actor i ((weights_dict, epsilon) tuple)
        Each actor has its own queue (maxsize=2) so the learner never blocks
        waiting for a slow actor to consume weights.

    algo_factory must be a module-level named callable (not a lambda) so it
    can be pickled for the child processes. Example:

        def make_algo():
            return DQN(DQNConfig(
                action_dim=2,
                network_factory=lambda: MLPQNetwork(5, 128, 2),
                ...
            ))

        runner = ParallelRunner(env_factory=..., algo=make_algo(), algo_factory=make_algo, ...)

    Note: network_factory inside make_algo CAN be a lambda — it only runs
    inside each child process and never needs to be pickled directly.
    """

    def __init__(
        self,
        env_factory,
        algo,
        algo_factory: Callable,
        config: RunnerConfig,
        num_actors: int = 6,
        updates_per_drain: int = 4,  # gradient steps per queue-drain cycle
        weight_sync_freq: int = 100,  # broadcast weights every N learner updates
        epsilon_base: float = 0.4,
        epsilon_alpha: float = 7.0,
        reward_shaper=None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        epsilon_base_decay: float = 1.0,  # multiply epsilon_base by this each weight sync
        epsilon_base_min: float = 0.01,  # floor for epsilon_base after decay
        lr_decay: float = 1.0,  # multiply current lr by this each weight sync
        lr_min: float = 1e-5,  # floor for lr after decay
        per_beta_increment: float = 0.0,  # add this to beta each weight sync (anneals toward 1.0)
        actor_random_warmup_steps: Optional[
            int
        ] = None,  # None -> fall back to cfg.train_start
    ):
        self.env_factory = env_factory
        self.algo = algo
        self.algo_factory = algo_factory
        self.config = config
        self.num_actors = num_actors
        self.updates_per_drain = updates_per_drain
        self.weight_sync_freq = weight_sync_freq
        self.epsilon_base = epsilon_base
        self.epsilon_alpha = epsilon_alpha
        self.reward_shaper = reward_shaper
        self.env_kwargs = env_kwargs or {}
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.epsilon_base_decay = epsilon_base_decay
        self.epsilon_base_min = epsilon_base_min
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.per_beta_increment = per_beta_increment
        self.actor_random_warmup_steps = actor_random_warmup_steps
        self.checkpointer = Checkpointer(config.ckpt_dir)

    def train(self):
        self.checkpointer.install_process_logger()
        self._train_impl()

    def _train_impl(self):
        """
        Launch actor processes and run the learner loop.

        The learner alternates between two phases in a tight loop:
          1. Drain: pull up to 512 transitions from the queue into the PER buffer.
             Episode boundaries (done=True) trigger checkpointing and logging.
          2. Update: run `updates_per_drain` gradient steps if the buffer is
             large enough. After `weight_sync_freq` updates, broadcast fresh
             weights and decayed epsilons to all actors.

        A daemon heartbeat thread prints debug state every 10 s so hangs
        can be diagnosed without attaching a debugger.
        """
        # Raise the open-file-descriptor limit; spawning N actors + queues can
        # hit the default limit (256 on macOS) when num_actors is large.
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 8192), hard))

        cfg = self.config
        actor_random_warmup_steps = (
            cfg.train_start
            if self.actor_random_warmup_steps is None
            else max(0, int(self.actor_random_warmup_steps))
        )

        # Restore learner state from disk if a previous run was checkpointed
        meta = self.checkpointer.load(self.algo)
        if meta:
            start_ep = meta["episode"] + 1
            total_steps = meta["total_steps"]
            best_score = meta.get("best_score", -1)
            best_avg100 = meta.get("best_avg100")
            # Clamp restored epsilon to the current configured floor so older
            # checkpoints cannot resume below the experiment's minimum.
            current_epsilon_base = max(
                self.epsilon_base_min,
                meta.get("epsilon_base", self.epsilon_base),
            )
            current_lr = meta.get("current_lr", self.algo.config.lr)
            current_beta = meta.get("current_beta", self.per_beta)
            best_avg100_text = (
                f"{best_avg100:.2f}" if best_avg100 is not None else "n/a"
            )
            print(
                f"Resuming from checkpoint — "
                f"episode={start_ep}  "
                f"total_steps={total_steps}  "
                f"peak_score={best_score}  best_avg100={best_avg100_text}  "
                f"epsilon_base={current_epsilon_base:.4f}  "
                f"lr={current_lr:.2e}  per_beta={current_beta:.4f}"
            )
            # Restore lr into the freshly-created optimizer (Adam state is not saved)
            self.algo.set_lr(current_lr)
        else:
            print("No checkpoint found — starting fresh.")
            start_ep = 1
            total_steps = 0
            best_score = -1
            best_avg100 = None
            current_epsilon_base = self.epsilon_base
            current_lr = self.algo.config.lr
            current_beta = self.per_beta
        actor_epsilons = _ape_x_epsilons(
            self.num_actors, current_epsilon_base, self.epsilon_alpha
        )
        print("Actor epsilons: " + "  ".join(f"{e:.4f}" for e in actor_epsilons))
        print(f"Actor random warmup steps: {actor_random_warmup_steps}")

        buffer = PrioritizedReplayBuffer(
            cfg.buffer_size, alpha=self.per_alpha, beta=self.per_beta
        )
        print(
            f"Using PrioritizedReplayBuffer (alpha={self.per_alpha}, beta={self.per_beta})"
        )

        buffer.beta = (
            current_beta  # restore annealed beta (or initial value on fresh start)
        )
        t0 = time.time()
        if buffer.load(cfg.ckpt_dir):
            print(
                f"Restored {len(buffer):,} transitions from replay buffer in {time.time() - t0:.1f}s."
            )
        else:
            print("No replay buffer checkpoint found — starting with empty buffer.")

        # 'spawn' context creates fresh interpreter per process — required for MLX
        # because Metal GPU contexts cannot be safely inherited by forked processes.
        ctx = mp.get_context("spawn")
        # maxsize=2 per actor queue: prevents the learner from queuing up stale
        # weight messages faster than actors can consume them.
        weight_queues = [ctx.Queue(maxsize=2) for _ in range(self.num_actors)]
        # Large enough to buffer bursts from all actors without dropping too many transitions
        transition_queue = ctx.Queue(maxsize=self.num_actors * 2_000)

        previous_boot_log_path = os.environ.get("MLX_RL_LOG_PATH")
        os.environ["MLX_RL_LOG_PATH"] = self.checkpointer.log_path

        actors = []
        initial_weights = self.algo.get_weights()
        for i in range(self.num_actors):
            p = ctx.Process(
                target=_actor_fn,
                args=(
                    i,
                    self.env_factory,
                    self.algo_factory,
                    self.reward_shaper,
                    weight_queues[i],
                    transition_queue,
                    actor_epsilons[i],
                    actor_random_warmup_steps,
                    self.env_kwargs,
                    initial_weights,
                    cfg.ckpt_dir,
                    bool(meta),  # skip_random_warmup — use policy immediately on resume
                ),
                daemon=True,  # actors die automatically if the learner exits
            )
            p.start()
            actors.append(p)

        print(f"Started {self.num_actors} actor processes.")

        learner_updates = 0
        episode = start_ep
        recent_scores: list = []  # rolling window (last 100) for mean score display
        is_best_score = False
        eval_states = None  # fixed set of states for Q-value diagnostics
        _recent_losses: list = []  # losses accumulated since last weight sync
        _recent_td_errors: list = []  # mean TD errors accumulated since last weight sync

        # Heartbeat: prints debug state every 10s so we can see where it's stuck
        # without the overhead of printing every iteration of the tight inner loop.
        _dbg = {"phase": "starting", "drained": 0, "updates": 0, "last_ep": episode}
        _stop_hb = threading.Event()

        def _heartbeat():
            while not _stop_hb.is_set():
                alive = sum(1 for p in actors if p.is_alive())
                print(
                    f"[HB] phase={_dbg['phase']}  "
                    f"buf={len(buffer)}  drained={_dbg['drained']}  "
                    f"updates={_dbg['updates']}  ep={_dbg['last_ep']}  "
                    f"actors_alive={alive}/{self.num_actors}"
                )
                _stop_hb.wait(10)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()

        try:
            while episode < start_ep + cfg.max_episodes:
                _dbg["phase"] = "drain"
                # Drain the transition queue in batches of up to 512 at a time.
                # Capping the drain prevents the learner from spending all its time
                # ingesting data and starving the gradient update step.
                drained = 0
                while drained < 512:
                    try:
                        s, a, r, ns, done, score = transition_queue.get_nowait()
                        buffer.push(s, a, r, ns, done)
                        total_steps += 1
                        drained += 1
                        _dbg["drained"] = drained

                        if done and score is not None:
                            # Track peak score separately from the checkpoint
                            # criterion, which is the rolling Avg100 once the
                            # window is fully populated.
                            is_best_score = score > best_score
                            if is_best_score:
                                best_score = score

                            recent_scores.append(score)
                            if len(recent_scores) > 100:
                                recent_scores.pop(0)  # keep only last 100

                            avg = sum(recent_scores) / len(recent_scores)
                            avg_ready = len(recent_scores) == 100
                            is_best = avg_ready and (
                                best_avg100 is None or avg > best_avg100
                            )
                            if is_best:
                                best_avg100 = avg

                            if episode % cfg.log_every == 0:
                                best_avg_text = (
                                    f"{best_avg100:6.1f}"
                                    if best_avg100 is not None
                                    else "   n/a"
                                )
                                print(
                                    f"Episode {episode:6d} | "
                                    f"Score: {score:4d} | "
                                    f"Avg100: {avg:6.1f} | "
                                    f"BestAvg100: {best_avg_text} | "
                                    f"Peak: {best_score:4d} | "
                                    f"Buf: {len(buffer):6d} | "
                                    f"Upd: {learner_updates}"
                                )

                            _dbg["phase"] = "checkpoint"
                            self.checkpointer.save(
                                self.algo,
                                meta={
                                    "episode": episode,
                                    "total_steps": total_steps,
                                    "best_score": best_score,
                                    "best_avg100": round(best_avg100, 4)
                                    if best_avg100 is not None
                                    else None,
                                    "epsilon_base": round(current_epsilon_base, 6),
                                    "current_lr": round(current_lr, 8),
                                    "current_beta": round(current_beta, 6),
                                },
                                is_best=is_best,
                                is_best_score=is_best_score,
                            )
                            episode += 1
                            _dbg["last_ep"] = episode
                            _dbg["phase"] = "drain"

                    except Exception:
                        break  # queue is empty — stop draining

                if len(buffer) >= cfg.train_start:
                    _dbg["phase"] = "update"

                    # Sample a fixed eval set once for Q-value diagnostics
                    if eval_states is None:
                        eval_batch, _, _ = buffer.sample(1000)
                        eval_states = eval_batch[0].copy()  # states only

                    for _ in range(self.updates_per_drain):
                        batch, tree_indices, is_weights = buffer.sample(cfg.batch_size)
                        loss, td_errors = self.algo.update(batch, weights=is_weights)
                        # Feed the fresh TD errors back into PER so priorities
                        # reflect the current network's surprisal, not stale values.
                        buffer.update_priorities(tree_indices, td_errors)
                        learner_updates += 1
                        _dbg["updates"] = learner_updates
                        _recent_losses.append(loss)
                        _recent_td_errors.append(float(np.mean(td_errors)))

                    # Periodically push latest weights (and decayed epsilons) to actors
                    if learner_updates % self.weight_sync_freq == 0:
                        _dbg["phase"] = "weight_sync"
                        # Decay epsilon_base toward epsilon_base_min so actors
                        # gradually become more greedy as the policy matures.
                        current_epsilon_base = max(
                            self.epsilon_base_min,
                            current_epsilon_base * self.epsilon_base_decay,
                        )
                        # Decay learning rate toward lr_min.
                        if self.lr_decay < 1.0:
                            current_lr = max(self.lr_min, current_lr * self.lr_decay)
                            self.algo.set_lr(current_lr)
                        # Anneal PER beta toward 1.0 for full IS correction.
                        if self.per_beta_increment > 0.0:
                            current_beta = min(
                                1.0, current_beta + self.per_beta_increment
                            )
                            buffer.beta = current_beta
                        actor_epsilons = _ape_x_epsilons(
                            self.num_actors, current_epsilon_base, self.epsilon_alpha
                        )
                        weights_np = (
                            self.algo.get_weights()
                        )  # numpy dict, safe to queue
                        for wq, eps in zip(weight_queues, actor_epsilons):
                            # Flush any unconsumed stale weight message before pushing
                            # the new one — maxsize=2 queues can still hold one stale msg.
                            while not wq.empty():
                                try:
                                    wq.get_nowait()
                                except Exception:
                                    break
                            try:
                                wq.put_nowait((weights_np, eps))
                            except Exception:
                                pass  # actor queue full — skip this sync cycle

                        # Q-value diagnostics — log mean Q, target Q, gap, loss, TD error
                        if eval_states is not None:
                            qs = self.algo.q_stats(eval_states)
                            mean_loss = (
                                np.mean(_recent_losses) if _recent_losses else 0.0
                            )
                            mean_td = (
                                np.mean(_recent_td_errors) if _recent_td_errors else 0.0
                            )
                            print(
                                f"[Q] upd={learner_updates}  "
                                f"q_online={qs['mean_q_online']:7.3f}  "
                                f"q_target={qs['mean_q_target']:7.3f}  "
                                f"gap={qs['q_gap']:+7.3f}  "
                                f"loss={mean_loss:.4f}  "
                                f"td={mean_td:.4f}  "
                                f"eps={current_epsilon_base:.4f}"
                            )
                            _recent_losses.clear()
                            _recent_td_errors.clear()

                    # Release GPU memory pool periodically to avoid fragmentation
                    if learner_updates % 50 == 0:
                        _dbg["phase"] = "clear_cache"
                        mx.clear_cache()
                else:
                    # Buffer not warm yet — yield CPU time so actors can fill it faster
                    _dbg["phase"] = "waiting_buffer"
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStopped by user. Latest checkpoint is saved.")
        finally:
            if previous_boot_log_path is None:
                os.environ.pop("MLX_RL_LOG_PATH", None)
            else:
                os.environ["MLX_RL_LOG_PATH"] = previous_boot_log_path
            _stop_hb.set()
            for p in actors:
                p.terminate()
            # join with timeout — p.join() can itself raise KeyboardInterrupt when
            # actors die noisily; we don't want that to abort the buffer save.
            for p in actors:
                try:
                    p.join(timeout=2)
                except Exception:
                    pass
            try:
                transition_queue.cancel_join_thread()
                transition_queue.close()
            except Exception:
                pass
            for wq in weight_queues:
                try:
                    wq.cancel_join_thread()
                    wq.close()
                except Exception:
                    pass

            print(f"Saving replay buffer ({len(buffer):,} transitions)...")
            t0 = time.time()
            buffer.save(cfg.ckpt_dir)
            print(f"Replay buffer saved in {time.time() - t0:.1f}s.")

        best_avg100_text = f"{best_avg100:.2f}" if best_avg100 is not None else "n/a"
        print(
            f"\nCycle complete. Peak score: {best_score} | "
            f"BestAvg100: {best_avg100_text}"
        )

    def test(
        self,
        best: bool = False,
        best_score: bool = False,
        env_kwargs_override: Optional[Dict[str, Any]] = None,
        num_episodes: int = 0,
        render: bool = True,
    ):
        self.checkpointer.install_process_logger()
        self._test_impl(
            best=best,
            best_score=best_score,
            env_kwargs_override=env_kwargs_override,
            num_episodes=num_episodes,
            render=render,
        )

    def _test_impl(
        self,
        best: bool = False,
        best_score: bool = False,
        env_kwargs_override: Optional[Dict[str, Any]] = None,
        num_episodes: int = 0,
        render: bool = True,
    ):
        """
        Run the greedy policy in a single process.

        No actors are spawned — inference runs directly in the learner process.
        Epsilon is 0 (always greedy).

        Args:
            num_episodes: Stop after this many episodes. 0 = run until Ctrl+C.
            render: Whether to render visually. Set False for fast batch evaluation.
        """
        if best_score:
            meta = self.checkpointer.load_best_score(self.algo)
            tag = "best_score"
        elif best:
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
            f"best_avg100: {meta.get('best_avg100', 'n/a')}  "
            f"peak_score: {meta.get('best_score', 'n/a')}"
        )

        test_kwargs = {**self.env_kwargs, **(env_kwargs_override or {})}

        scores = []
        episode = 0
        try:
            while True:
                env = self.env_factory(render_mode=render, **test_kwargs)
                state = env.reset()
                episode += 1
                while True:
                    action = self.algo.select_action(state)
                    state, _, done, info = env.step(action)
                    if done:
                        score = info["score"]
                        scores.append(score)
                        print(f"Episode {episode} — Score: {score}")
                        break
                env.close()
                if num_episodes > 0 and episode >= num_episodes:
                    break
        except KeyboardInterrupt:
            pass

        if len(scores) > 1:
            arr = np.array(scores)
            print(
                f"\n--- Results over {len(arr)} episodes ---"
                f"\n  Mean:   {arr.mean():.1f}"
                f"\n  Std:    {arr.std():.1f}"
                f"\n  Median: {np.median(arr):.1f}"
                f"\n  Min:    {arr.min()}"
                f"\n  Max:    {arr.max()}"
            )
