# Training Notes — Lessons Learned

Everything discovered through empirical runs on Breakout (pixel observations).
Ordered from most impactful to least.

---

## 1. Use Double DQN, not standard DQN

**Problem:** Standard DQN systematically overestimates Q-values. The target uses
`max_a Q_target(s', a)`, which always picks the highest predicted value. Estimation
noise means this maximum is an overestimate in expectation. The bias compounds over
time — overestimated Q-values lead to a policy that is confident but wrong, which
then generates bad experience that makes estimates worse.

**Symptom:** Training curve peaks (Avg100 ~16) then collapses back to ~8–9 and
never recovers, even after hundreds more episodes. The collapse is not random —
it happens reliably because overestimation is systematic.

**Fix:** `DoubleDQN` (already in `algorithms/double_dqn.py`). Online net selects
the best next action; target net evaluates it. Because selection and evaluation
use different networks with uncorrelated noise, the same noise spike cannot inflate
both simultaneously.

```python
# experiments/breakout_cnn_dqn.py
from algorithms.double_dqn import DoubleDQN
def make_algo():
    return DoubleDQN(DQNConfig(...))
```

---

## 2. Do NOT clip rewards for a single-game setup

**Common advice:** Clip rewards to [-1, 1]. This is correct when training one
model across 49 Atari games (the original multi-game setup) to normalize
the reward scale across games with wildly different score magnitudes.

**Why it backfires here:** Breakout brick values are [1, 1, 4, 4, 7, 7] by row.
With clipping, every brick break gives reward=1 — the agent has no gradient signal
distinguishing a 1-point bottom-row brick from a 7-point top-row brick. It learns
to break whatever is easiest to reach and stops developing the skill to reach higher
rows. BestAvg100 plateaued at 9.7 vs 11.3+ without clipping.

**Rule:** Only clip rewards when training across multiple games with different score
scales. For a single game, use raw rewards — they encode useful difficulty gradients.

---

## 3. Start training only after the buffer has meaningful coverage

**Problem:** `train_start=5_000` on a 1M buffer means learning begins when the
buffer is 0.5% full. The first thousands of gradient updates train on a tiny,
unrepresentative sample. The model learns biased weights from early random-policy
experience. When the buffer later fills with diverse data, those weights have to
unlearn, causing a performance dip that it may never fully recover from.

**Symptom:** Performance peaks early (ep ~150–200) then regresses to below the peak
and stays there. Seen consistently across all runs regardless of algorithm.

**Fix:** `train_start=50_000` — the original benchmark ratio of 5% buffer fill before
learning begins. With 8 actors this takes longer wall-clock time but the foundation
is far more stable.

```python
config=RunnerConfig(
    buffer_size=1_000_000,
    train_start=50_000,   # was 5_000 — fill 5% of buffer before first update
    ...
)
```

---

## 4. Store pixel observations as uint8, not float32

**Problem:** Storing (4, 84, 84) frames as float32 costs 220 KB per transition.
A 1M buffer would require ~210 GB — impossible on any single machine.

**Fix:** Store frames as uint8 (0–255). Cost drops to 55 KB per transition.
A 1M buffer costs ~53 GB.

The normalization (÷255) happens once inside `CNNQNetwork.__call__()` on the GPU,
not in Python per-transition:

```python
# algorithms/dqn.py — CNNQNetwork.__call__
x = x.astype(mx.float32) * (1.0 / 255.0)   # uint8 → float32 on GPU
x = mx.transpose(x, (0, 2, 3, 1))
...
```

The buffer's `sample()` methods must not force `dtype=np.float32` on states —
drop the explicit dtype so the stored dtype is preserved:

```python
np.array(states),          # was: np.array(states, dtype=np.float32)
np.array(next_states),     # was: np.array(next_states, dtype=np.float32)
```

**Memory budget:**
| Buffer | RAM (uint8) |
|--------|-------------|
| 100k   | ~5.5 GB     |
| 500k   | ~28 GB      |
| 1M     | ~53 GB      |
| 2M     | ~106 GB ⚠️  |

---

## 5. Add a life-loss penalty

**Problem:** The only reward signal is brick value (1/4/7). Losing a life gives
reward=0. Early in training the agent has no feedback that missing the ball is bad
— it can't yet break enough bricks to see the downstream cost of a lost life.

**Fix:** Subtract 1.0 from reward on each life loss. This is a dense signal
available from the very first episode — the agent just has to miss once to receive
it. It bootstraps ball-tracking behavior before the brick gradient becomes useful.

```python
# envs/breakout/env.py — _advance_ball()
if self.ball_y > SCREEN_HEIGHT:
    self.lives -= 1
    reward -= 1.0   # life-loss penalty
    ...
```

Do not add more shaping beyond this. The brick value gradient already handles
row-targeting incentives. Per-step survival penalties and height bonuses are
redundant given raw brick rewards.

---

## 6. Buffer size: 1M matches the original DQN setup

The original 2015 DQN paper used a 1M transition replay buffer for all Atari games.
With uint8 storage (see §4), this costs ~53 GB.

Smaller buffers (100k) cause the model to overfit to recent experience — older
high-quality transitions get evicted before the model can learn from them fully.
The 1M buffer ensures diverse experience remains available throughout training.

---

## 7. Reward clipping is a multi-game normalisation trick, not a general rule

The original DQN paper clipped rewards because it trained one model on 49 different
games. Pong rewards ±1, Breakout rewards up to 7, other games score in hundreds.
Clipping made one learning rate work across all of them.

For a single game you lose information. The relative magnitude of rewards within
a game is signal, not noise.

---

## 8. 256 actors requires 256 machines (or don't bother)

The distributed training setup (2018) ran 256 actors on 256 separate CPU machines, each with
dedicated CPU/RAM, connected to a single GPU learner over a network queue.

On a single machine:
- 8 actors: well-matched to available CPU cores, leaves headroom for learner
- 12–16 actors: probably the ceiling before CPU context-switching overhead hurts
- 256 actors: RAM fits (~40 GB for processes alone) but many processes/core means
  the actors run slower than 8 would, netting less throughput

The right lever for more data on one machine is **longer training**, not more actors.

---

## 9. The original DQN (2015) specs

For reference — this is what was achievable in 2015 on a single machine:

- **Hardware:** Single NVIDIA GPU (GTX Titan era), single machine, no distribution
- **Training:** 50 million frames per game, ~38 days per game
- **Buffer:** 1M transitions (float32 at the time — ~220 GB, used RAM+disk probably)
- **Batch size:** 32
- **Optimizer:** RMSProp (lr=0.00025)
- **Epsilon:** Annealed 1.0 → 0.1 over first 1M frames, then fixed at 0.1
- **Train start:** 50k random frames before first update
- **Target update:** Hard copy every 10k gradient steps
- **Games trained:** 49 Atari games, each independently

The main gap is total frame
count — they ran 50M frames, most runs here are in the low millions.

---

## 10. PER beta should ideally anneal

`per_beta=0.4` is fixed in the current config. The PER paper (Schaul et al., 2015)
recommends annealing beta from 0.4 → 1.0 over training. At beta=1.0, importance
sampling fully corrects the bias introduced by prioritised sampling. At beta=0.4,
it only partially corrects it.

This is a low-priority fix — the model learns despite it — but a beta schedule
would improve correctness as training matures.

---

## 11. Log interleaving is normal, not a training bug

Running `python -m experiments.breakout_cnn_dqn --test` while training is in
progress causes the test process to write to the same `train.log` via `_TeeStream`.
The "Loaded best checkpoint" and "Episode 1 — Score: X" lines that appear mid-training
are from the test process, not a bug in the training loop. They do not affect training.

---

## 12. train_start=50k eliminates early collapse (confirmed)

**Evidence from Run 4** (DoubleDQN, raw rewards, life penalty, train_start=50k):
- Peak score reached **284** — highest of any DoubleDQN run (vs 63–68 with train_start=5k)
- No catastrophic collapse after the peak — Avg100 oscillates 7–12 rather than sliding to 7 and staying
- BestAvg100 held at 12.0 for 1100+ episodes without dropping

Previous runs with train_start=5k all showed the same pattern: peak around ep 150–200 then a clear downward trend. This run has no downward trend — just noise.

**Conclusion:** train_start=50k is confirmed correct. Keep it.

---

## 13. Fixed epsilon floods the buffer with stale exploration data

**Observation:** After ep 750 (peak), BestAvg100 stuck at 12.0 through ep 1878 despite 190k more gradient updates. The policy is not consolidating its gains.

**Why:** `epsilon_base_decay=1.0` means actor epsilons never change (0.4 → 0.01 fixed forever). As the policy improves, the high-epsilon actors (especially actor 0 at ε=0.4) still play 40% randomly, continuously filling the 1M buffer with low-quality random-policy transitions. The learner trains on this diluted data and cannot consistently exploit what it has learned.

**Fix:** Add epsilon decay so actors gradually shift toward exploitation as training matures:

```python
epsilon_base_decay=0.9999,   # multiply epsilon_base each weight sync
epsilon_base_min=0.05,       # floor — never go fully greedy
```

At weight_sync_freq=200 updates, decay of 0.9999 per sync means:
- After 100k updates (500 syncs): base = 0.4 × 0.9999^500 ≈ 0.36
- After 500k updates (2500 syncs): base = 0.4 × 0.9999^2500 ≈ 0.20
- After 2M updates (10k syncs): base ≈ 0.05 (floor)

This mirrors the benchmark epsilon annealing (1.0 → 0.1 over 1M frames) adapted for the distributed multi-actor setup.

---

## 14. Persist epsilon_base across restarts

**Problem:** Every Ctrl+C + restart reset `epsilon_base` back to 0.4, losing all decay progress. A run that had decayed to ε≈0.23 after 1.1M updates would restart at 0.4 and waste thousands of updates re-exploring before recovering.

**Fix:** Save `epsilon_base` in the checkpoint metadata and restore it on resume:

```python
# saved in meta dict every episode:
"epsilon_base": round(current_epsilon_base, 6)

# restored on resume:
current_epsilon_base = meta.get("epsilon_base", self.epsilon_base)
```

The resume log now shows the restored value:
```
Resuming from checkpoint — episode=1689 ... epsilon_base=0.3585
```

The `meta.get(..., self.epsilon_base)` fallback means old checkpoints (without the key) still load cleanly.

---

## 15. Save and restore the replay buffer on shutdown

**Problem:** Every restart discards the entire replay buffer and refills from scratch. The real cost is not the 4-minute warmup to `train_start=50k` — it's that the learner trains for hours on a small, low-diversity buffer instead of the 800k+ transitions accumulated before the restart. This caused Avg100 to dip to 7–8 after every restart before slowly recovering.

**Fix:** Save the buffer to disk on Ctrl+C, restore it on startup.

- `PrioritizedReplayBuffer.save(ckpt_dir)` — writes all data arrays + SumTree priorities to `replay_buffer_tmp.npz`, then atomically renames to `replay_buffer.npz`. The tmp→rename pattern means a crash mid-save never corrupts the previous good file.
- `PrioritizedReplayBuffer.load(ckpt_dir)` — restores data and SumTree in exact slot positions, preserving priorities.
- Buffer save happens in the `finally` block after actors are terminated.
- With a full 1M buffer (~47 GB), save and load each take ~8 seconds.

**Evidence:** With buffer restore in place, training resumed at ep 1689 with 205k transitions already in the buffer and jumped straight to `update` phase — no warmup delay at all.

---

## 16. Actor subprocess cleanup noise on Ctrl+C is harmless

The long `KeyboardInterrupt` tracebacks from `SpawnProcess-1` through `SpawnProcess-8` that appear after Ctrl+C are the actor child processes printing their own cleanup noise to stderr. They do not indicate a bug. The learner process completes its `finally` block (saves checkpoint + buffer) and exits cleanly regardless.

The tracebacks come from Python's multiprocessing queue feeder threads trying to flush when interrupted — an upstream Python/multiprocessing issue with no clean fix.

---

## 17. epsilon_base_min=0.05 is too low — exploration collapses

**Observation:** With `epsilon_base_decay=0.9999` and `epsilon_base_min=0.05`, epsilon_base decayed from 0.4 → ~0.08 over ~15k episodes. At 0.08, the distributed spread formula makes 7 of 8 actors essentially greedy (epsilons < 0.01). Only actor 0 retains any meaningful exploration.

**Symptom:** BestAvg100 stuck at 14.3 for 10,600+ episodes (ep 6950–17596). The buffer saturated at ep ~7736 (~1M transitions), and after that the buffer was being filled almost entirely with greedy-policy rollouts. A greedy policy that is stuck produces a feedback loop: low-diversity buffer → no new signal → policy can't improve → buffer stays low-diversity. Extreme case: `Score=0` at ep 17550 — agent stopped tracking the ball entirely (policy: "don't move").

**Fix:** `epsilon_base_min=0.2`. At base=0.2, actor 0 still explores at 20%, and the spread keeps actors 1–4 at meaningful non-zero epsilons. The floor is high enough that the buffer always contains diverse experience.

```python
epsilon_base_min=0.2   # was 0.05
```

---

## 18. lr reduction mid-training stabilizes but doesn't break plateaus

**Observation (ep ~15537):** Avg100 drifted down to 7.4 (new low) over several thousand episodes after buffer fully cycled. Manually changed lr from 1e-4 → 3e-5.

**Effect:** Stopped the downward drift (Avg100 stabilized ~8–11) but did not reverse the trend or break the 14.3 BestAvg100 ceiling. The root cause (epsilon collapse) was still present.

**Conclusion:** lr reduction alone is not sufficient when the buffer is starved of diverse experience. The correct fix is to address epsilon collapse first (§17), then use a lr schedule to complement it.

---

## 19. Three-pronged fix for long plateaus: exploration + lr schedule + PER beta annealing

When BestAvg100 has been stuck for thousands of episodes, apply all three together:

**1. Raise epsilon_base_min to 0.2** — restores actor diversity (see §17).

**2. Add a lr schedule** — start 1e-4, decay multiplicatively each weight_sync, floor at 1e-5:
```python
lr_decay=0.9998,   # 1e-4 → 1e-5 over ~11,500 syncs (2.3M updates)
lr_min=1e-5,
```
Adam optimizer state resets on each restart (not saved), so the schedule restarts from `config.lr`. `current_lr` is saved in checkpoint meta so the schedule continues across restarts within a run.

**3. Anneal PER beta 0.4 → 1.0** — at beta=1.0, importance-sampling fully corrects the bias from prioritised sampling. Fixed beta=0.4 permanently under-corrects. Increment per weight_sync:
```python
per_beta_increment=0.0001,   # reaches 1.0 from 0.4 in 6,000 syncs (1.2M updates)
```
`current_beta` is saved in checkpoint meta and restored (same pattern as `epsilon_base`).

The resume log now shows all three: `epsilon_base=...  lr=...  per_beta=...`

---

## 20. MSE loss + no gradient clipping caused every catastrophic forgetting episode

**Root cause of the "peak then collapse" pattern seen in all early runs:**

The loss function was MSE: `loss = mean(weights * td²)`. When a TD error is large (common early in training when Q-values are random), the gradient scales quadratically. A TD error of 5.0 produces a gradient 25× larger than a TD error of 1.0. This caused the network to overwrite its learned weights on any batch containing large errors.

No gradient clipping meant that even after Huber loss was applied per-sample, a batch of many transitions with moderate errors could still sum to a very large global gradient vector.

**Fix (applied in `algorithms/dqn.py`):**

1. **Huber loss** — quadratic for |td| ≤ 1, linear beyond:
   ```python
   huber = mx.where(abs_td <= 1.0, 0.5 * td ** 2, abs_td - 0.5)
   return mx.mean(weights * huber), abs_td
   ```
   This is what the original paper called "error clipping" — they clipped TD errors to [-1, 1] before squaring, which is mathematically equivalent.

2. **Gradient clipping** (max_norm=10.0):
   ```python
   grads, _ = optim.clip_grad_norm(grads, max_norm=10.0)
   ```
   Rescales the entire gradient vector if its L2 norm exceeds 10.0. Prevents any single bad batch from making a catastrophically large parameter update.

**Evidence:** All runs prior to this fix showed the same pattern regardless of target_update_freq, reward clipping, or autolaunch settings. The policy would peak (Avg100 ~12–16) then regress to ~8 and never recover. Both fixes together address this at the implementation level.

---

## 21. Frame skip = 4 (action repeat)

Added `frame_skip=4` to `BreakoutEnv` — each call to `step()` repeats the action for 4 physics frames internally and returns the accumulated reward and the final frame.

**Why this matters:**
- Each agent decision covers 4 frames instead of 1 — the effective problem is simpler (fewer decisions per ball crossing)
- Q-value targets are more stable: the horizon is 4× shorter in decision steps
- 4× environment throughput for free — actors generate the same number of transitions per wall-clock second but each covers more game time
- This is standard in the original 2015 DQN and was applied to all 49 Atari games

**Implementation details:**
- `frame_skip: int = 4` constructor param on `BreakoutEnv` (default=4, so existing non-pixel uses are unaffected)
- Physics loop runs `frame_skip` times per `step()`, rewards accumulate, breaks early on `done`
- Frame capture happens once after all 4 physics steps (no max-pooling needed — our env has no sprite flickering unlike ALE)
- `env_kwargs={"obs_type": "pixels", "frame_skip": 4}` in experiment config passes through to all 8 actor subprocesses correctly
- `reward_shaper` clips the accumulated reward (possibly from multiple brick hits) to [-1, 1] — consistent with the benchmark's per-decision-step clipping

**Paddle/ball dynamics with frame_skip=4:**
- Paddle moves 32px per decision (8px × 4 frames) — crosses 512px screen in ~14 decisions
- Ball moves ~20px per decision (speed=5 × 4 frames) — crosses screen height in ~25 decisions
- Paddle is fast enough to track the ball

**Requires fresh start** — existing checkpoints and buffer were collected at 1-frame granularity. Mixing 1-frame and 4-frame transitions corrupts training. Delete checkpoints before restarting.

**Performance evidence:**
- Without frame skip (all prior runs): BestAvg100 peaked at **25.3** after 2.4M updates
- With frame skip=4: BestAvg100 reached **93.0** after only **287k updates**

This is a 3.7× score improvement using 8.4× fewer gradient updates. The dramatic gap confirms that frame skip is the single most impactful optimization applied in this project. The key mechanism: without frame skip, consecutive observations are near-identical (ball moves 5px) and the agent is making decisions on redundant information. With frame skip=4, the ball moves ~20px between decisions — each observation is visually distinct and each transition carries real information about ball trajectory. This makes every sample in the replay buffer significantly more valuable.

---

## 22. LR decay hurts fine-tuning from a checkpoint; actors need checkpoint weights at startup

**Two discoveries from the restart experiments:**

### 22a. Actors start with random weights on resume (bug, now fixed)

When resuming from a checkpoint, the learner loads the checkpoint weights correctly, but actor subprocesses started with `total_steps=0`. The actor loop has:
```python
if random.random() < epsilon or total_steps < train_start:
    action = random.randint(0, env.action_dim - 1)
```
This caused all 8 actors to play **completely randomly** for their first 50,000 steps each — even though the learner had good weights loaded. The 50k warmup buffer filled with random-policy experience, negating the purpose of loading from a checkpoint.

**Fix:** Pass `skip_random_warmup=True` to actors on resume, setting `total_steps = train_start` at startup so they immediately use the policy:
```python
total_steps = train_start if skip_random_warmup else 0
```
**Evidence:** Before fix, training Avg100 after resume was 7–13 (random play). After fix, immediately 136–141 (ep-8579 policy in action).

### 22b. LR decay + small buffer on resume causes policy drift

Restarting from the best checkpoint (ep 8579, BestAvg100=136.6) with a fresh empty buffer and lr=1e-4 caused the policy to degrade immediately after training began (Avg100 dropped from 141 → 60–80 within 150k updates).

**Root cause:** The 50k warmup fills the buffer with a narrow distribution from one specific policy (the checkpoint policy). Training aggressively (lr=1e-4) on this small, concentrated buffer causes the Q-network to overfit that distribution. Once actors receive updated weights and start producing different data, the policy drifts.

**In the original from-scratch run this didn't matter:** the buffer naturally filled with diverse data from gradually improving policies. On resume you start with a good but narrow distribution.

**Conclusion:** LR decay was added to handle late-training instability, but it also causes the LR to be very low (~4.9e-5) at the time of the best checkpoint. Resetting to 1e-4 on resume is too aggressive for a small buffer. For fine-tuning restarts, either:
- Use lr ~5e-5 (match the decayed value at the checkpoint) until the buffer fills, then raise
- Or train from scratch with fixed LR and fixed epsilon (original paper style)

### 22c. Decision: train from scratch, no LR decay, no epsilon decay (since revised)

The original DQN used fixed lr=1e-4 and fixed epsilon throughout all 50M frames. Both our LR decay and epsilon decay were added empirically to fight symptoms (late-training instability, exploration collapse) that may themselves be artifacts of the small-buffer / short-run conditions.

This resulted in a fixed-epsilon from-scratch run (epsilon=0.4 throughout), which reached BestAvg100=38.9/ball but plateaued there for ~5000 episodes. The plateau was diagnosed as a measurement ceiling from epsilon noise, not a learning failure (see §25). Epsilon decay was subsequently re-enabled with more conservative settings (see §26).

Config at time of this decision:
```python
epsilon_base_decay=1.0,   # no decay
epsilon_base_min=0.4,     # fixed at 0.4
lr_decay=1.0,             # no decay
lr_min=1e-4,              # fixed at 1e-4
```

---

## 23. 300k buffer produces a lower ceiling than 1M (empirically confirmed)

**Experiment:** Reduced `buffer_size` from 1M → 300k to test whether faster data turnover helps break through plateaus.

**Result:** BestAvg100 peaked at **110.6** vs **180.2** with 1M buffer. Same training setup otherwise.

**Why:** Two mechanisms compound:
1. **Rare transitions evicted too fast.** High-value "tunnel drilling" episodes (ball behind bricks) are rare. With 300k buffer and 8 actors at epsilon=0.4 generating ~800 transitions/episode, the buffer turns over completely every ~375 episodes. Rare transitions are evicted before the learner can sample them enough times to consolidate the associated Q-values.
2. **Epsilon noise floods a small buffer.** With epsilon_base=0.4, actors fill the buffer with noisy exploratory experience. In a 1M buffer this noise is diluted by good prior transitions. In a 300k buffer it overwhelms them.

**Also:** train_start=50k becomes proportionally too large for a 300k buffer (16.7% vs 5% for 1M). This caused ~20 BestAvg100 points of slower early learning before the buffer wrapped.

**Conclusion:** The original paper's reasoning for 1M was correct. Reverted to 1M.

---

## 24. Life-loss as terminal state (single-ball episodes)

The original DQN training trick: treat each life loss as `done=True`, so each ball is its own episode. The game resets on every ball drop.

**Before:** `done=True` only when all 5 lives are exhausted. Episode = full game (up to 5 balls, high variance in length and score).

**After:** `done=True` on every life loss. Episode = one ball. Runner calls `reset()` after each.

```python
# envs/breakout/env.py — _advance_ball()
if self.ball_y > SCREEN_HEIGHT:
    self.lives -= 1
    reward -= 1.0
    done = True  # DeepMind trick: each life is its own episode
    break
```

**Benefits:**
- Much shorter, uniform episodes (~10–25 decisions each vs up to 125 for 5-life game)
- Cleaner credit assignment — agent maximizes reward per ball, not per game
- BestAvg100 becomes a per-ball metric. Multiply by 5 to get approximate game-equivalent.
- Eliminates the multi-life variance that was causing inconsistent training signal

**Performance:** With single-ball episodes + epsilon decay, the agent broke BestAvg100=51.5/ball by ~576k updates. With fixed epsilon (no decay), it plateaued at 38.9 for ~5000 episodes before epsilon decay was re-enabled (see §25, §26).

**Note:** The checkpoint directory mixed old multi-life checkpoints with new single-ball ones. The `--test --best` flag loaded the old multi-life checkpoint (ep 8579, best_avg100=136.61) since its metric was higher. Use `--test` (no `--best`) to evaluate the current run's latest checkpoint.

---

## 25. Fixed epsilon=0.4 creates a measurement ceiling, not a learning ceiling

**Observation:** After the from-scratch single-ball run, BestAvg100 plateaued at **38.9** for ~5000 episodes despite the agent clearly being capable of much more — single-ball scores of 337, 224, 181, 178 appeared regularly during the same plateau period.

**Root cause:** With epsilon_base fixed at 0.4 forever, every 100-episode window in the training log is heavily polluted by noisy high-epsilon episodes. The BestAvg100 metric can only improve if a 100-episode window happens to contain enough low-epsilon (capable) episodes to pull the average above 38.9. The fixed exploration floor makes this rare.

This is a **measurement problem**, not a learning problem. The agent's greedy policy is substantially better than 38.9/ball.

**Evidence from test run:** Running `--test --best` showed individual episode scores of 220, 227, 227, 160, 140, 132, 125, 121, 121, 110 — far above the 38.9 training average. (Note: this loaded the old multi-life checkpoint, not the current run, due to checkpoint mixing — but the point stands.)

**Fix:** Re-enable epsilon decay so actors gradually shift toward exploitation as training matures, allowing BestAvg100 to reflect improving policy quality.

---

## 26. Epsilon decay re-enabled (second attempt, more conservative)

**Context:** Section 17 documented that epsilon_base_min=0.05 with decay=0.9999 caused exploration collapse (~15k episodes). Section 22c then switched to fixed epsilon=0.4. Now re-enabling with a higher floor.

**Previous failure:** decay=0.9999, min=0.05 → epsilon hit the floor too fast, 7/8 actors became near-greedy, buffer saturated with greedy-policy rollouts, policy froze.

**New settings:**
```python
epsilon_base_decay=0.9995,   # faster per-sync decay (reaches floor ~1.5M updates)
epsilon_base_min=0.05,       # same floor, but reevaluated with 1M buffer context
```

**Why this might work better now:**
- 1M buffer (vs smaller buffers in section 17 experiments) provides much larger diversity cushion before exploration collapse can set in
- Single-ball episodes are shorter — actors generate more episodes per wall-clock time, so the buffer stays more diverse even as epsilon falls
- The plateau at 38.9 is severe enough that the risk of re-enabling decay is worth it

**Watch for:** If Avg100 drops to near-zero (score=0 episodes appearing regularly) after epsilon_base approaches 0.05, that's the same collapse as §17. If so, raise epsilon_base_min to 0.1–0.2.

**Result (confirmed working):** BestAvg100 broke through the 38.9 ceiling at ep ~23850 and reached **51.5** by ep ~24800 (~576k updates). This confirmed the plateau in §25 was a measurement artifact from fixed epsilon, not a learning failure. Epsilon decay with 1M buffer did not cause exploration collapse.

---

## 27. The benchmark epsilon schedule vs our distributed adaptation

**Original DQN (2015):**
- Linear anneal: 1.0 → 0.1 over first 1M frames (~250k decisions with frame_skip=4)
- Fixed at 0.1 for remainder of training (up to 50M frames)
- Single actor, so epsilon was a single scalar

**Our distributed adaptation:**
- `epsilon_base` sets the base of the actor spread formula
- Actor epsilons: `epsilon_i = epsilon_base^(1 + alpha * i/(N-1))` with alpha=4.0, N=8
- At start (epsilon_base=0.4): actors range 0.4000 → 0.2370 → 0.1404 → ... → 0.0102
- As epsilon_base decays, the whole distribution shifts toward exploitation

**Benchmark-equivalent translation:**
- Their 0.1 floor maps to roughly `epsilon_base_min=0.1` here
- Their 250k-decision anneal maps to `epsilon_base_decay=0.999` (floor in ~300 syncs / ~60k updates)
- We use slower decay (0.9995) and lower floor (0.05) — less aggressive than the benchmark

**Key insight:** The original paper started at epsilon=1.0 and used the first 1M frames as both warmup and exploration. We start at 0.4 (lower) because our train_start=50k means the random warmup is already built-in — we don't need the full 1.0 starting point.

---

## 28. Checkpoint directory mixing breaks --test --best

**Problem:** After switching from multi-life to single-ball episodes, the checkpoint directory still contained the old multi-life checkpoints. The `--test --best` flag selects the checkpoint with the highest `best_avg100`. Since the old multi-life run had `best_avg100=136.61` and the new single-ball run only reached `best_avg100=38.9` at that point, `--test --best` loaded the wrong checkpoint.

**Symptom:** `Loaded best checkpoint — trained for 8579 episodes, best_avg100: 136.61` appeared when expecting the current run's checkpoint. The test scores (~60–70/ball) were from the old policy, not the current one.

**Fix:** Delete the checkpoint directory before starting a new run with a different episode structure:
```bash
rm -rf checkpoints/breakout_cnn_dqn
```

**Rule:** Any time the episode structure changes (multi-life ↔ single-ball, reward scaling, frame skip), clear checkpoints. The `best_avg100` values are not comparable across different episode structures, and the wrong checkpoint will silently be selected.

---

## 29. Late-stage collapse after BestAvg100=60: isolate PER beta first

**Observation (Mar 20 run):**
- BestAvg100 reached **60.0**, then training regressed for thousands of episodes.
- Avg100 drifted from ~40s down into the ~20s, with occasional high-score spikes (for example 150-200+) still appearing.
- Replay stayed full at 1M while heartbeat frequently showed very small drain batches (`drained=1..6`) but the learner still performed fixed update batches.

**Key diagnostics:**
- At collapse time, learner updates were much higher than actor environment steps (high replay pressure).
- Checkpoint metadata showed `current_beta` reached **1.0** late in run, while the best-Avg100 checkpoint was earlier at lower beta (~0.80).

**Interpretation:**
- This pattern is more consistent with late-stage instability from learner/replay imbalance (and possibly aggressive PER correction at beta=1.0) than with `batch_size=32` alone.

**Changes made (single-step ablations):**
```python
epsilon_alpha=4.0          # reverted spread from 1.0 back to prior setting
epsilon_base_decay=0.9997  # slower decay than 0.9995
per_beta_increment=0.0     # freeze beta annealing (was 1e-4)
```

**Rationale:** keep ablations controlled and change one dominant instability source first (`beta` drift to 1.0), rather than changing multiple throughput knobs at once.

---

## 30. `train_start=50k` in this code means global gate + per-actor forced-random warmup

**Important distinction:**
- Learner starts updates when `len(buffer) >= train_start` (global replay warmup).
- Each actor also forces random actions while its own `total_steps < train_start`.

**Consequence with 8 actors:**
- Up to roughly `8 * 50k = 400k` forced-random actor steps can be generated.
- Those transitions are not discarded; they enter replay and are sampled once learner updates begin after the first 50k.

**Takeaway:**
- A 50k global warmup is normal for distributed style training.
- Combining that with 50k per-actor forced-random warmup is much stronger exploration pressure and can overfill replay with low-quality early data.

---

## 31. Decouple actor random warmup from learner `train_start`

**Problem:**
- `train_start=50_000` was being used for two different gates:
1. Learner start condition (`len(buffer) >= train_start`) — global replay warmup.
2. Actor forced-random condition (`total_steps < train_start`) — per-actor warmup.
- With 8 actors, this produced up to ~`8 * 50k = 400k` forced-random transitions, far beyond the intended global 50k warmup.

**Code change:**
- Added a dedicated runner parameter: `actor_random_warmup_steps` (default `None` means keep old behavior via `cfg.train_start`).
- Actor loop now uses `total_steps < random_warmup_steps` instead of `total_steps < train_start`.
- Breakout experiment now sets:
```python
train_start=50_000              # learner global warmup gate
actor_random_warmup_steps=6_250 # per-actor forced random (~50k total across 8 actors)
```

**Why:**
- Preserve the stabilizing effect of a 50k replay warmup for the learner.
- Avoid over-injecting low-quality random transitions early in training.
- Keep exploration primarily from the distributed epsilon spread rather than a long forced-random phase.

**Operational note:**
- This takes effect only after restarting the training process.

---

## 32. Reduce learner throughput before changing batch size

**Context:**
- After the actor warmup fix, training recovered strongly (BestAvg100 rose into the ~50-60 range) but still showed late-stage downward drift / low-30s plateaus.
- The cleaner next ablation is learner throughput, not `batch_size`, because `updates_per_drain` directly changes replay consumption without changing gradient batch statistics.

**Change:**
```python
updates_per_drain=2   # was 4
batch_size=32         # unchanged
```

**Why this ablation first:**
- With `batch_size=32`, reducing `updates_per_drain` from 4 -> 2 cuts replay samples consumed per drain cycle from `128` to `64`.
- This reduces learner pressure on a full replay buffer while keeping optimizer noise characteristics the same.
- Changing `batch_size` at the same time would confound two effects:
  - samples consumed per cycle
  - gradient noise / optimization dynamics

**Decision:**
- Hold `batch_size=32` constant for this run.
- If late-stage drift still persists after this change, then test `batch_size=64` as a separate ablation.

---

## 33. Ball must reset to paddle when a wall is cleared

**Bug discovered (2026-03-21):**
When wall 1 was cleared, `_build_wall()` rebuilt the bricks but `_attach_ball()` was never called. The ball kept moving at its current velocity directly into the new wall. It would get trapped above the bricks, bouncing around the high-value rows (7+7 pts each) indefinitely until it eventually fell through, racking up hundreds of free points.

**Symptom:** Sudden score spikes like Peak=790 during training. BestAvg100 results looked artificially high — the agent was exploiting a physics bug, not learning real Breakout.

**Fix:**
```python
# envs/breakout/env.py — in _advance_ball(), after _build_wall():
if not self.bricks:
    self.walls_cleared += 1
    if self.walls_cleared >= MAX_WALLS:
        done = True
    else:
        self.bricks = self._build_wall()
        self._attach_ball()   # ← added
    break
```

**Impact:** All BestAvg100 results before this fix are invalid — they included free above-wall points. The real post-fix ceiling starts from scratch. First legitimate best: **BestAvg100=56.58**.

**Rule:** This matches real Atari Breakout (and the original ALE): clearing a wall resets the ball to the paddle and the player must FIRE to begin the next wall.

---

## 34. Post-peak collapse: root cause is Q-value overestimation, not just epsilon diversity

**Observation (multiple runs):**
Training consistently peaked at BestAvg100 ~55-56 around 420-550k updates, then collapsed to Avg100=20-30 by 620-730k updates — regardless of epsilon floor (0.2 or 0.3).

**What we tried that didn't work:**
- `epsilon_base_min=0.2 → 0.3` (avg floor epsilon 5.4% → 9.3%): same peak, same collapse
- Resuming from best checkpoint with empty buffer + higher epsilon: policy degraded immediately

**Root cause (from the distributed training paper):**
With `target_update_freq=10,000`, the online network ran 10,000 gradient updates before the target network caught up. The `max` operator in Bellman targets always selects the action with the highest noisy Q-value estimate, systematically overestimating returns. Over 10k steps without correction, this bias compounds:
1. Online Q-values drift upward beyond true returns
2. Policy acts on inflated estimates → takes bad actions → gets low reward
3. Buffer fills with bad transitions → Q-values crash → collapse

**Fix:**
```python
target_update_freq=100   # was 10_000 — matches Ape-X paper (every 100 training batches)
```

**Why 100:**
The distributed training paper (Horgan et al. 2018) explicitly updates the target network every 100 training batches. This keeps the target tracking the online network closely, limiting the window for overestimation to accumulate. Double DQN reduces overestimation at action *selection*; frequent target updates reduces it at *evaluation*.

---

## 35. Match the benchmark epsilon schedule: start=1.0, decay to floor in ~250k updates

**Original DQN (2015):** Single actor, linear epsilon 1.0 → 0.1 over 1M frames = ~250k gradient updates, fixed at 0.1 forever after.

**Our previous config:** `epsilon_base=0.4` → avg starting epsilon across 8 actors was only ~12.1%, barely above the benchmark's floor. The entire high-exploration phase was skipped.

**Fix:**
```python
epsilon_base=1.0          # all actors start fully random (1^anything = 1)
epsilon_base_decay=0.9990 # reaches floor in ~250k updates (applied every 200 updates)
epsilon_base_min=0.3      # avg floor ~9.3% across 8 actors ≈ DeepMind's 10%
```

**How decay timing works:**
- Decay is applied every `weight_sync_freq=200` updates (i.e. 1,250 applications to span 250k updates)
- `1.0 × 0.9990^1250 ≈ 0.28` → hits floor at ~250k updates
- Previous `decay=0.9997` took ~800k updates to reach 0.3 — floor hit way too late

**Epsilon at floor with alpha=4.0, N=8, base=0.3:**
```
actor 0: 30.0%   actor 4: 4.2%
actor 1: 18.3%   actor 5: 2.5%
actor 2: 11.2%   actor 6: 1.5%
actor 3:  6.8%   actor 7: 0.2%
avg: ~9.3%   ← matches DeepMind's 10%
```

---

## 36. target_update_freq=100 is too frequent — use 2500 to match the benchmark

**Problem with freq=100:**
With `target_update_freq=100` and `weight_sync_freq=200`, the target network syncs twice per log interval. The `gap` diagnostic always shows `+0.000` because by the time we log, the target has just caught up. More importantly, the online network is effectively chasing its own tail — there is no stable target signal.

**Observed symptom:**
At ~340k updates with freq=100, loss jumped from 0.09 → 0.35 and mean TD error exploded from 0.63 → 4.0 while Q-values dropped (3.8 → 2.6). The target was syncing so often it provided no stability — equivalent to online Q-learning with no target network.

**Why the distributed "100 batches" doesn't directly translate:**
The distributed training paper (Horgan et al. 2018) uses 360 actors. With that many actors, 100 training batches still corresponds to many thousands of environment steps. With only 8 actors, 100 gradient updates is a tiny window — far less environment experience than the distributed setup's 100 batches.

**Fix:**
```python
target_update_freq=2500   # matches DeepMind: 10,000 frames / frame_skip=4 = 2,500 gradient updates
```

**Why 2500:**
The original DQN used a hard target update every 10,000 frames. With frame_skip=4, that's 2,500 gradient updates. This gives the online network 2,500 steps to learn before the target catches up — enough stability to train on, without so much lag that the target diverges dangerously.

**Gap metric is now useful:**
With freq=2500 > weight_sync_freq=200, the gap will show a non-zero value between target syncs. A healthy training run should show a small stable gap. A widening gap indicates overestimation building up.

---

## 37. epsilon_base=1.0 start hurts multi-actor training — distributed diversity ≠ single-agent diversity

**Experiment:** Changed epsilon schedule to match the benchmark: `epsilon_base=1.0`, `decay=0.9990`, `floor=0.3`. Result: BestAvg100 only reached **38.0** vs **56.6** with `epsilon_base=0.4/floor=0.2`.

**Root causes:**

1. **Starting at 1.0 with 8 actors floods the buffer with random garbage**
   The first ~250k updates, all 8 actors play nearly randomly. These ~500k low-quality transitions fill the buffer and cycle through during the critical 250-500k update learning window. The policy tries to learn from data dominated by near-random play.

2. **Higher floor (0.3) means permanently noisier buffer**
   At floor, avg epsilon across actors is 9.3% (vs 5.4% with floor=0.2). Actor 0 is doing 30% random actions constantly. More noise in the buffer = lower data quality ceiling.

3. **The 1.0 start was for a single actor — the distributed setup doesn't need it**
   The original paper started at epsilon=1.0 to ensure diverse early exploration from a single actor. With 8 actors in a distributed spread from epsilon=0.4 down to ~0.01, diversity is already baked in via the spread. Starting at 1.0 adds no diversity benefit — it just pollutes the buffer.

**Rule:** With the distributed multi-actor setup, epsilon diversity comes from the **spread across actors**, not from high starting epsilon. Keep `epsilon_base=0.4` so even the most exploratory actor isn't playing randomly.

**Best config for epsilon:**
```python
epsilon_base=0.4        # actor 0 starts at 40%, actor 7 at ~1%
epsilon_base_decay=0.9997  # decays to floor at ~462k updates
epsilon_base_min=0.2    # avg floor ~5.4% (vs DeepMind's 10%, but data quality > diversity)
```

---

## 38. Q-value diagnostic logging

Added `[Q]` log lines every `weight_sync_freq` (200) updates showing:
```
[Q] upd=N  q_online=X.XXX  q_target=X.XXX  gap=±X.XXX  loss=X.XXXX  td=X.XXXX  eps=X.XXXX
```

**What each metric means:**
- `q_online` / `q_target`: mean max Q-value on fixed 1000-state eval set for online/target networks
- `gap`: `q_online - q_target`. Healthy: small (±0.03). Danger: gap growing consistently upward = overestimation
- `loss`: mean Huber loss over last 200 updates. Healthy: stable. Danger: rising while Avg100 falls
- `td`: mean absolute TD error. Healthy: 0.2-0.4. Danger: exploding (4.0+ = instability)

**Observed healthy baseline (target_update_freq=2500):**
- gap oscillates ±0.01-0.03, never drifts
- loss stable ~0.026-0.040
- td stable ~0.27-0.35

**Observed unhealthy (target_update_freq=100):**
- gap always +0.000 (target too fresh, no separation)
- loss jumped to 0.35, td exploded to 4.0 at 340k updates

---

## 39. target_update_freq revised to 10,000 gradient updates

**Previous setting:** `target_update_freq=2500` (section 36 — derived as the original 10,000 frames / frame_skip=4).

**Revised to:** `target_update_freq=10_000`

**Why the 2,500 derivation was wrong for our setup:**
The original 10,000-frame schedule assumed a single actor where every frame corresponds to one env step and roughly one gradient update. In our distributed setup with 8 actors, env steps accumulate much faster than gradient updates. Dividing by frame_skip=4 was a frame→update conversion that doesn't apply the same way here.

Using 10,000 gradient updates gives the target network more lag, which produces a more stable training signal. The `gap` metric (q_online − q_target) now has room to be meaningfully non-zero between syncs, making it a useful diagnostic again.

```python
target_update_freq=10_000   # hard copy online → target every 10k gradient updates
```

**What to watch:** A slowly widening gap over many target intervals indicates Q-value overestimation building. A stable ±0.01–0.05 gap is healthy.

---

## 40. PER beta annealing: 0.4 → 1.0 over 1.2M gradient updates

**Setup:**
```python
per_beta=0.4,              # initial IS-weight exponent
per_beta_increment=0.0001, # added once per weight sync
weight_sync_freq=200,      # syncs per 200 gradient updates
```

Beta reaches 1.0 after **6,000 weight syncs = 1,200,000 gradient updates** — covering the full training run.

**Why this schedule:**
- **Beta < 1 early:** Tolerate bias while the value function is still noisy. High-variance gradients from full IS correction early on hurt more than the bias.
- **Beta → 1 late:** As Q-values stabilize, correct the sampling bias fully for accurate learning. PER without IS correction converges to a biased policy.

**Implementation detail:** Beta is incremented once per `weight_sync_freq` step (not per env step or gradient update), and is saved/restored in checkpoints so annealing continues correctly across restarts.

**Confirmed working:** `current_beta` appears in checkpoint JSON and is logged in the status line.

---

## 41. Current training configuration (2026-03-22 fresh cold-start run)

Full parameter snapshot for the run that achieved **BestAvg100=85.8, Peak=832** at ep ~100k.

### Algorithm
```python
DoubleDQN(
    DQNConfig(
        action_dim=4,
        network_factory=lambda: CNNQNetwork(obs_shape=(4,84,84), action_dim=4),
        lr=1e-4,
        gamma=0.99,
        target_update_freq=10_000,   # hard copy online → target every 10k gradient updates
    )
)
```

### Runner
```python
ParallelRunner(
    num_actors=8,
    updates_per_drain=1,             # 1 gradient update per actor drain cycle
    weight_sync_freq=200,            # push online weights to actors every 200 updates
    epsilon_alpha=4.0,               # Ape-X spread exponent
    epsilon_base=0.4,                # initial per-actor epsilon base
    epsilon_base_decay=0.9997,       # decays once per weight sync (every 200 updates)
    epsilon_base_min=0.35,           # raised from 0.2 at ep ~100k (see §42); floor reached after ~1,000 weight syncs = 200k updates
    per_alpha=0.6,                   # PER priority exponent
    per_beta=0.4,                    # initial IS-weight exponent
    per_beta_increment=0.0001,       # increments per weight sync → reaches 1.0 after 6,000 syncs = 1.2M updates
    lr_decay=1.0,                    # no LR decay
    lr_min=1e-4,
    actor_random_warmup_steps=6_250, # each actor plays randomly for 6,250 steps before using policy
)
```

### RunnerConfig
```python
RunnerConfig(
    buffer_size=1_000_000,
    batch_size=64,
    train_start=50_000,              # learner waits until buffer has 50k transitions
    max_episodes=1_000_000,
    render_every=200,
    log_every=50,
)
```

### Environment
```python
env_kwargs={"obs_type": "pixels", "frame_skip": 4}
reward_shaper: np.clip(reward, -1.0, 1.0)
```

### Actor epsilon spread at epsilon_base=0.4
```
Actor 0 (most explore): epsilon = 0.4^1         = 0.4000
Actor 1:                epsilon = 0.4^1.571      = 0.2370
Actor 2:                epsilon = 0.4^2.143      = 0.1404
Actor 3:                epsilon = 0.4^2.714      = 0.0832
Actor 4:                epsilon = 0.4^3.286      = 0.0493
Actor 5:                epsilon = 0.4^3.857      = 0.0292
Actor 6:                epsilon = 0.4^4.429      = 0.0173
Actor 7 (most greedy):  epsilon = 0.4^5         = 0.0102
```

At original floor (epsilon_base=0.2): mean ε ≈ 4.2% — too greedy, below the benchmark's training ε of 10%.
At revised floor (epsilon_base=0.35): mean ε ≈ 9.6% — matches the benchmark's training epsilon.

| Floor | Actor 0 | Actor 7 | Mean ε |
|-------|---------|---------|--------|
| 0.20 (original) | 0.200 | 0.00032 | ~4.2% |
| 0.35 (revised)  | 0.350 | 0.00525 | ~9.6% |
| Original DQN (single agent) | — | — | 10% |
| Distributed paper (360 actors) | — | — | ~6.3% |

### Run trajectory
| Milestone | Episode | Updates | BestAvg100 | Notes |
|-----------|---------|---------|------------|-------|
| Training starts | ~4,550 | 27 | 1.5 | buf hit 50k |
| First lucky spike | 12,239 | 37,490 | 4.83 | Peak=345, immediate collapse |
| Learning begins | ~42,000 | ~225k | 5.3 | Avg100 leaves 1.x range |
| Epsilon floor hit | ~53,000 | ~640k | — | eps=0.2000 |
| New record | 53,254 | 640k | 61.1 | Peak=412, greedy test scored 782 |
| New record | ~60,900 | ~1,113k | 82.4 | Peak=662 |
| New record | ~75,350 | ~1,976k | 83.9 | Peak=814 |
| New record | ~77,650 | ~2,114k | **85.8** | Peak=832 |
| epsilon_base_min raised | ~100,600 | ~3,435k | 85.8 | Changed 0.2→0.35 to match the benchmark's 10% training ε |

### Q-value diagnostics at ep ~100k (before epsilon change)
```
q_online ≈ 2.60–2.66  q_target ≈ 2.604–2.631  gap = ±0.03–0.05  loss = 0.005–0.006  td = 0.27–0.29
```
Q values had plateaued/oscillated in 2.49–2.57 range for ~600k updates. Identified root cause:
mean ε at floor=0.2 was only ~4.2%, well below the benchmark's 10% training epsilon → under-exploration.

### Greedy test results (epsilon=0.05, full game, 5 lives)
At ep ~94,850: **mean=333.7, std=139.8, median=376, max=588** over 35 episodes.
Original DQN benchmark: ~401. We were at ~83% of the benchmark under identical eval conditions.

---

## 42. epsilon_base_min=0.2 was too low — mean ε fell below the benchmark's training epsilon

At ep ~100k the run plateaued at BestAvg100=85.8 for ~25,000 episodes. Q values oscillated
in the 2.49–2.57 range without net growth for ~600k updates.

**Root cause:** with `epsilon_base_min=0.2` and `epsilon_alpha=4.0`, the mean actor epsilon
at the floor was only ~4.2% — significantly below the benchmark's training epsilon of 10%.

| Setup | Mean training ε |
|-------|----------------|
| Original DQN (single agent) | 10% |
| Distributed paper (360 actors) | ~6.3% |
| Ours at floor=0.2 | ~4.2% ← too greedy |
| Ours at floor=0.35 | ~9.6% ← matches the benchmark |

**Fix applied at ep ~100,600:** raised `epsilon_base_min` from `0.2` → `0.35`.

The checkpoint was backed up first (`breakout_cnn_dqn_backup_ep100k`). The trained weights
were kept — only the exploration floor changed. The replay buffer was intentionally NOT saved
(~55GB, impractical) and refreshes naturally in ~6,000 episodes.

On resume, the `max(epsilon_base_min, saved_epsilon_base)` clamp in the runner immediately
applied the new floor — no code changes needed beyond the config value.

**Expected effect:** richer exploration → buffer fills with more diverse experience →
policy discovers strategies it was too greedy to explore before → BestAvg100 should
eventually break past 85.8.

---

## 43. Fresh cold-start run with epsilon_base_min=0.35 (started 2026-03-22)

**Setup:** Completely fresh run (no checkpoint loaded, empty buffer) with `epsilon_base_min=0.35`.
Previous backup at `checkpoints/breakout_cnn_dqn_backup_ep100k` (BestAvg100=85.8) was retained.

**Motivation:** After the continued run (ep 100k–142k) failed to break past 85.8 even with the
new epsilon floor, a fresh cold-start was chosen to avoid any stale buffer bias from the old
under-explored policy. A clean buffer ensures the richer exploration fills replay from scratch
with diverse, high-epsilon experience.

**Parameters (unchanged from §41 except epsilon floor):**
```python
epsilon_alpha=4.0,
epsilon_base=0.4,
epsilon_base_decay=0.9997,
epsilon_base_min=0.35,   # was 0.2 in §41
```

### Run trajectory (fresh run)
| Milestone | Episode | Updates | BestAvg100 | Notes |
|-----------|---------|---------|------------|-------|
| Training starts | ~4,550 | ~27 | — | buf hit 50k |
| Buffer full | ~55,500 | ~889k | 48.2 | Buf=1M, Peak=449 |
| Steady learning | ~57,800 | ~1,003k | 48.3 | All 8 actors alive, eps=0.3500 |

### Comparison vs previous run at same stage
| Episode | Prev run (floor=0.2, mean ε≈4.2%) | This run (floor=0.35, mean ε≈9.6%) |
|---------|------------------------------------|--------------------------------------|
| ~53k | BestAvg100=61.1 (epsilon floor hit) | — |
| ~57.8k | ~61–82 range | BestAvg100=48.3 |

Behind in early BestAvg100 — expected. Higher exploration floor means actors take more random
actions, lowering the rolling average vs the more exploitative old run. The hypothesis is that
the richer, more diverse buffer will allow the policy to discover strategies that the old run's
greedy actors missed, eventually pushing past 85.8.

### Q-value diagnostics at ep ~57,800
```
q_online ≈ 1.73–1.78  q_target ≈ 1.731  gap = +0.00–+0.045  loss = 0.0016–0.0022  td = 0.12
```
Target net just synced at upd=1,000,000. Q online already pulling ahead again. Healthy.

---

## 44. floor=0.35 run stalled at 57.5; switched to floor=0.27 fresh run (2026-03-23)

### Why floor=0.35 was abandoned
The floor=0.35 run (§43) stalled at BestAvg100=57.5 from ep ~60,550 onward (~9,750 episodes /
~450k updates without improvement). Greedy eval showed mean=390.3 over 62 episodes — the policy
was genuinely strong, but the high epsilon floor (mean ε≈10.5%) was too noisy for BestAvg100 to
reflect it. The rolling 100-episode average was dragged down by random actions in every episode.

### eval_states fix (parallel_runner.py)
Discovered that `eval_states` (the 1,000-state fixed set used for `[Q]` logging) was re-sampled
fresh from the buffer on every restart. This made Q values incomparable across restarts — jumps
from ~2.2 → ~4.3 on reload were pure measurement artifacts, not real overestimation.

**Fix:** `eval_states` is now saved to `checkpoints/.../eval_states.npy` on shutdown (alongside
the buffer) and restored on startup. Q values are now consistent across restarts.

### epsilon_base_min tuning
Mean ε comparison across all runs:

| Floor | Mean ε | Outcome |
|-------|--------|---------|
| 0.20 | ~4.2% | Peaked 85.8, stalled 25k eps, under-explored |
| 0.27 | ~6.4% | Current run — matches the distributed paper's ~6.3% |
| 0.35 | ~10.5% | Stalled at 57.5, metric too noisy |

### Run trajectory (floor=0.27, fresh cold-start, 2026-03-23)
| Milestone | Episode | Updates | BestAvg100 | Notes |
|-----------|---------|---------|------------|-------|
| Training starts | ~4,550 | ~27 | — | buf hit 50k |
| Epsilon floor hit | ~51,800 | ~710k | 41.7 | eps=0.2700 |
| Floor hit + jump | ~53,200 | ~796k | 57.8 | Rapid climb after floor |
| Plateau begins | ~60,550 | ~1,154k | 57.5→74.3 | Slow climb via render evals |
| Q value plateau | ~80,000 | ~2,100k | ~74 | Q oscillating 2.37–2.59 for 3M+ updates |
| Current state | ~136,450 | ~5,028k | **74.3** | Peak=807, Q≈2.55, stalled |

### Greedy eval at ep ~53k
```
Mean: 422.0  Std: 142.6  Median: 424.5  Min: 118  Max: 802  (52 episodes, epsilon=0.05)
```
**Exceeds Double DQN benchmark of 401.2.** Policy is strong; BestAvg100 metric understates it
due to epsilon noise in training episodes.

### Why tunnel strategy hasn't been learned consistently
Peak=807 proves the tunnel has been discovered occasionally. However consistent tunneling hasn't
emerged at 5M gradient updates. Key reasons:
- The original training used ε=1.0→0.1 annealing (more early random exploration to stumble into tunnel)
- The original training did ~50M gradient updates vs our 5M (10× more reinforcement of the behavior)
- Tunnel requires sustained directed play over many steps — hard credit assignment
- Q values flat-lined for 3M updates — policy stuck in "break reachable bricks" local optimum

**Checkpoint backed up:** `checkpoints/breakout_cnn_dqn_backup_ep136k` (BestAvg100=74.3, Peak=807)

---

## 45. Reward shaping + full 5-life episodes experiment (2026-03-23)

**Motivation:** Agent at ep ~136k has not learned the tunnel strategy consistently despite
Peak=807 showing it's possible. Root cause: short single-life episodes (terminal_on_life_loss=True)
limit credit assignment horizon, and the flat reward signal gives no gradient toward the tunnel.

### Changes made

**1. Normalised brick rewards (`experiments/breakout_cnn_dqn.py`)**

Raw brick values: 1, 1, 4, 4, 7, 7 (bottom → top rows).
Normalised by dividing by 7 (max value):

| Row | Raw | Normalised |
|-----|-----|-----------|
| Top (rows 1-2) | 7 | 1.000 |
| Mid (rows 3-4) | 4 | 0.571 |
| Bot (rows 5-6) | 1 | 0.143 |

Preserves the relative gradient (top bricks still worth 7× bottom) while keeping rewards in [0, 1].

**2. Life-loss penalty: -0.05 per life lost**

Small enough to avoid PER over-prioritisation (which caused collapse in earlier runs with -1.0),
large enough to signal "this was bad". Ratio: losing a life ≈ 1/3 of a bottom brick.

**3. Full 5-life episodes (`terminal_on_life_loss=False`)**

Previously each life was its own episode (~200-500 steps). Now each episode spans all 5 lives
(~1,000-2,500 steps). Benefits:
- Life-loss penalty is a mid-episode signal — agent continues after losing a life and can
  learn the consequence
- Longer credit assignment horizon — tunnel payoff can be experienced within one episode
- More diverse buffer per episode (5× more game states covered)

`actor_random_warmup_steps=6_250` and `train_start=50_000` unchanged — both measured in
environment steps (transitions), not episodes, so the 8×6,250=50k ratio still holds.

**4. `life_lost` flag added to `BreakoutEnv` (`envs/breakout/env.py`)**

```python
self.life_lost = False          # reset at start of each step()
# set to True in _advance_ball() when ball_y > SCREEN_HEIGHT
```
Allows the reward shaper to detect life loss without tracking state externally.

### New reward shaper
```python
def reward_shaper(env, reward, done):
    shaped = reward / 7.0
    if env.life_lost:
        shaped -= 0.05
    return shaped
```

### Updated env_kwargs
```python
env_kwargs={"obs_type": "pixels", "frame_skip": 4, "terminal_on_life_loss": False},
```

### DQN paper comparison
- Original DQN (Mnih 2015): clipped rewards [-1, 1], terminal_on_life_loss=True, single agent
- Ours: normalised rewards [0, 1] with gradient intact, terminal_on_life_loss=False, 8 actors
- Key difference: our agent will see the full-game arc and experience life-loss consequences

**Fresh cold-start run with all changes above + epsilon_base_min=0.27.**

---

## Hardware reference

| Component | Memory |
|-----------|--------|
| 1M uint8 pixel buffer | ~53 GB |
| 8 actor processes | ~1.5 GB |
| Learner + Metal GPU heap | ~2 GB |
| macOS overhead | ~10 GB |
| **Total** | **~66 GB** |
| **Free headroom** | **~62 GB** |
