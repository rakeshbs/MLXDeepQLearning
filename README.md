# MLX Reinforcement Learning

Small reinforcement learning playground built around custom game environments and MLX-based DQN variants.

Current environments:
- `Flappy Bird`
- `Breakout`

Current training setup:
- `DQN`
- `DoubleDQN`
- `Prioritized Experience Replay`
- `ParallelRunner` with distributed actor/learner layout

## Requirements

This project is written for Python 3.10 and uses `MLX`, so it is intended for Apple Silicon machines.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mlx
pip install -r requirements.txt
```

`requirements.txt` currently contains the non-MLX runtime dependencies:
- `pygame`
- `numpy`

## Project Layout

```text
algorithms/
  base.py
  buffers.py
  dqn.py
  double_dqn.py
envs/
  base.py
  flappy_bird/
  breakout/
experiments/
  breakout/
    cnn_dqn.py        ← Breakout pixel CNN experiment
    notes.md          ← Training notes and lessons learned
  flappy/
    dqn.py            ← Flappy Bird state-vector DQN
    double_dqn.py     ← Flappy Bird state-vector Double DQN
    cnn_dqn.py        ← Flappy Bird pixel CNN experiment
training/
  runner.py
  parallel_runner.py
  checkpoint.py
play.py
```

## Environments

### Flappy Bird

`envs/flappy_bird/env.py`

Observation modes:
- `state`: 5-dim vector
- `pixels`: 4 stacked `84x84` grayscale frames

Action space:
- `0`: do nothing
- `1`: flap

### Breakout

`envs/breakout/env.py`

Atari-inspired rules:
- `5` lives
- `2` brick walls per game
- max score `864`
- brick values by row: `7, 7, 4, 4, 1, 1`

Observation modes:
- `state`: 8-dim vector
- `pixels`: 4 stacked `84x84` grayscale frames

Action space:
- `0`: NOOP
- `1`: FIRE
- `2`: RIGHT
- `3`: LEFT

## Training Experiments

### Breakout

Train:

```bash
python -m experiments.breakout.cnn_dqn
```

Test latest checkpoint:

```bash
python -m experiments.breakout.cnn_dqn --test
```

Test best checkpoint (by Avg100):

```bash
python -m experiments.breakout.cnn_dqn --test --best
```

Test best single-episode score checkpoint:

```bash
python -m experiments.breakout.cnn_dqn --test --best-score
```

Test with full 5-life game (no terminal on life loss):

```bash
python -m experiments.breakout.cnn_dqn --test --best --full-game
```

Test with epsilon-greedy evaluation (e.g. ε=0.05):

```bash
python -m experiments.breakout.cnn_dqn --test --best --epsilon=0.05
```

Test without rendering (faster, for benchmarking):

```bash
python -m experiments.breakout.cnn_dqn --test --best --no-render
```

Run a fixed number of test episodes:

```bash
python -m experiments.breakout.cnn_dqn --test --best --episodes=100
```

Flags can be combined freely:

```bash
python -m experiments.breakout.cnn_dqn --test --best --full-game --no-render --episodes=100 --epsilon=0.05
```

### Flappy Bird

State-vector DQN:

```bash
python -m experiments.flappy.dqn
python -m experiments.flappy.dqn --test
python -m experiments.flappy.dqn --test --best
```

State-vector Double DQN:

```bash
python -m experiments.flappy.double_dqn
python -m experiments.flappy.double_dqn --test
python -m experiments.flappy.double_dqn --test --best
```

Pixel-based CNN Double DQN:

```bash
python -m experiments.flappy.cnn_dqn
python -m experiments.flappy.cnn_dqn --test
python -m experiments.flappy.cnn_dqn --test --best
```

## Manual Play

Run Breakout:

```bash
python play.py --game breakout
```

Run Flappy Bird:

```bash
python play.py --game flappy
```

Controls:

Breakout:
- `Space`: serve
- `A` / `D` or `Left` / `Right`: move paddle
- `R`: reset
- `Esc` or `Q`: quit

Flappy Bird:
- `Space`: flap
- `R`: reset
- `Esc` or `Q`: quit

## Checkpoints

Checkpoints are written under `checkpoints/<experiment_name>/` as:
- `latest.npz` / `latest.json`
- `best.npz` / `best.json`

For the parallel runner:
- `latest` is always the most recent learner state
- `best` is selected using the best rolling `Avg100`, not the best single episode

## Notes

- The pixel experiments are much heavier than the state-vector ones.
- `ParallelRunner` uses multiple actor processes, so repeated `pygame` startup lines during launch are normal.
- You may see a `pkg_resources` deprecation warning from `pygame`; that is upstream package noise, not a project-specific error.
