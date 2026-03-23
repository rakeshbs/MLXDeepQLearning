from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .base import BaseAlgorithm


@dataclass
class DQNConfig:
    action_dim: int
    network_factory: Callable[[], nn.Module]  # returns a fresh Q-network
    lr: float = 1e-3
    gamma: float = 0.99
    target_update_freq: int = 1_000  # hard-copy online → target every N updates


class MLPQNetwork(nn.Module):
    """
    3-layer MLP: flat state vector → hidden → hidden → Q-values per action.

    Suitable for low-dimensional state observations (e.g. the 5-dim Flappy Bird
    state vector). Each output neuron corresponds to one discrete action's Q-value.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)  # no activation on output — raw Q-values


class CNNQNetwork(nn.Module):
    """
    Conv stack followed by an MLP head.

    Standard Atari CNN architecture.
    Expects input shape (batch, channels, height, width) — e.g. (B, 4, 84, 84).
    MLX Conv2d uses (B, H, W, C) internally, so the channel dimension is
    transposed at the start of the forward pass.
    """

    def __init__(self, obs_shape: tuple, action_dim: int):
        super().__init__()
        channels, h, w = obs_shape
        # Three conv layers with progressively smaller kernels / strides
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute flattened feature size analytically so the Linear head
        # can be sized correctly at construction time without a dummy forward pass.
        # Formula: floor((spatial_dim - kernel) / stride) + 1
        ch = ((h - 8) // 4 + 1)
        cw = ((w - 8) // 4 + 1)
        ch = ((ch - 4) // 2 + 1)
        cw = ((cw - 4) // 2 + 1)
        ch = ((ch - 3) // 1 + 1)
        cw = ((cw - 3) // 1 + 1)
        flat = 64 * ch * cw  # 64 output channels × spatial dimensions

        self.fc1 = nn.Linear(flat, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def __call__(self, x):
        # x arrives as (B, C, H, W); MLX Conv2d expects (B, H, W, C)
        # Normalise uint8 observations [0, 255] → float32 [0, 1] here so the
        # buffer can store raw uint8 pixels at 4× lower memory cost.
        x = x.astype(mx.float32) * (1.0 / 255.0)
        x = mx.transpose(x, (0, 2, 3, 1))
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # flatten spatial dimensions
        x = nn.relu(self.fc1(x))
        return self.fc2(x)  # raw Q-values, no activation


# Keep old name as alias so existing code that imports QNetwork still works.
QNetwork = MLPQNetwork


def _loss_fn(model, states, actions, targets, weights):
    """
    Weighted MSE loss between Q(s, a) and the Bellman target.

    Selecting Q values by integer action index (fancy indexing) avoids
    computing gradients for the non-chosen actions, which is correct —
    only the taken action has a known target.

    IS weights from PER are applied per-sample before the mean so that
    high-priority (frequently-replayed) transitions do not dominate.
    """
    q = model(states)
    # Index into Q-values with the action taken: shape (batch,)
    q_selected = q[mx.arange(len(actions)), actions]
    td = q_selected - targets
    # Compute Huber loss with correct gradients in both regions:
    #   |td| <= 1: 0.5 * td^2  → gradient = td
    #   |td|  > 1: |td| - 0.5  → gradient = sign(td)
    # abs_td must NOT have stop_gradient here — the linear region needs
    # gradients to flow through abs(td). A separate stop_gradient copy is
    # returned for PER priority updates, which must not affect the loss gradient.
    abs_td = mx.abs(td)
    huber = mx.where(abs_td <= 1.0, 0.5 * td ** 2, abs_td - 0.5)
    return mx.mean(weights * huber), mx.stop_gradient(abs_td)


class DQN(BaseAlgorithm):
    """
    Standard Deep Q-Network (Mnih et al., 2015).

    Bellman target: r + γ * max_a[ Q_target(s', a) ]

    Two networks are maintained:
      - online: trained every update step.
      - target: a periodic hard copy of online, providing stable learning
                targets. Without it, both the prediction and the target shift
                every step, causing oscillations or divergence.
    """

    def __init__(self, config: DQNConfig):
        self.config = config
        self.online = config.network_factory()
        self.target = config.network_factory()
        # Materialise parameters before the first sync — MLX is lazily evaluated,
        # so without mx.eval() the parameters are not yet allocated on the GPU.
        mx.eval(self.online.parameters())
        self._sync_hard()
        self.optimizer = optim.Adam(learning_rate=config.lr)
        # Fuse loss + gradient computation into a single MLX graph for efficiency
        self.loss_and_grad = nn.value_and_grad(self.online, _loss_fn)
        self._update_count = 0

    def select_action(self, state: np.ndarray) -> int:
        # Add a batch dimension (1, *state_shape) before passing through the network
        s = mx.array(state[None])
        q = self.online(s)
        mx.eval(q)  # force materialisation so .item() can read the result
        return int(mx.argmax(q[0]).item())

    def update(self, batch: tuple, weights=None) -> tuple:
        states, actions, rewards, next_states, dones = batch
        s  = mx.array(states)
        a  = mx.array(actions)
        r  = mx.array(rewards)
        ns = mx.array(next_states)
        d  = mx.array(dones)
        # Default to uniform weights when PER IS weights are not provided
        w  = mx.array(weights) if weights is not None else mx.ones(len(states))

        targets = self._compute_targets(r, ns, d)
        # loss_and_grad returns ((loss, abs_td_errors), gradients) in one pass
        (loss, td_errors), grads = self.loss_and_grad(self.online, s, a, targets, w)
        grads, _ = optim.clip_grad_norm(grads, max_norm=10.0)
        self.optimizer.update(self.online, grads)
        # CRITICAL: optimizer.state must be included in mx.eval(). Adam maintains
        # moment arrays (m, v) as lazy MLX computations. If they are never eval'd,
        # each update appends to an ever-growing lazy graph, eventually causing a
        # GPU OOM crash. Evaluating them here materialises and resets the chain.
        mx.eval(self.online.parameters(), self.optimizer.state, loss, td_errors)

        self._update_count += 1
        if self._update_count % self.config.target_update_freq == 0:
            self._sync_hard()
        return loss.item(), np.array(td_errors)

    def q_stats(self, states: np.ndarray) -> dict:
        """
        Compute Q-value diagnostics on a fixed evaluation set.

        Returns mean max-Q for online and target networks and their gap.
        A steadily rising mean_q_online with flat/falling performance indicates
        overestimation. A large gap between online and target indicates the
        target is lagging behind a drifting online network.
        """
        s = mx.array(states)
        q_online = self.online(s)
        q_target = self.target(s)
        mx.eval(q_online, q_target)
        mean_q_online = float(mx.mean(mx.max(q_online, axis=1)).item())
        mean_q_target = float(mx.mean(mx.max(q_target, axis=1)).item())
        return {
            "mean_q_online": mean_q_online,
            "mean_q_target": mean_q_target,
            "q_gap": mean_q_online - mean_q_target,
        }

    def _compute_targets(self, rewards, next_states, dones):
        """
        Standard DQN: target net selects AND evaluates the best next action.

        max_a Q_target(s', a) always picks the highest predicted Q-value,
        which introduces an upward bias (overestimation). DoubleDQN fixes this
        by separating selection (online) from evaluation (target).
        """
        next_q    = self.target(next_states)
        max_next_q = mx.max(next_q, axis=1)
        # (1 - done) masks out the future-return term for terminal transitions
        targets   = rewards + self.config.gamma * max_next_q * (1.0 - dones)
        mx.eval(targets)  # materialise before returning so gradients don't flow through targets
        return targets

    def _sync_hard(self):
        """
        Full parameter copy: online → target.

        Called at initialisation (so both nets start identically) and every
        target_update_freq gradient steps. A hard copy is simpler than Polyak
        averaging and works well when target_update_freq is large enough.
        """
        self.target.update(self.online.parameters())
        mx.eval(self.target.parameters())

    def set_lr(self, lr: float) -> None:
        self.optimizer.learning_rate = lr

    def get_weights(self) -> dict:
        # tree_flatten converts the nested parameter dict into a flat (key, array) list
        # so it can be shipped through a multiprocessing Queue as plain numpy arrays.
        return {k: np.array(v) for k, v in tree_flatten(self.online.parameters())}

    def set_weights(self, weights: dict) -> None:
        # Convert numpy arrays back to MLX before loading into the network
        weights_mx = {k: mx.array(v) for k, v in weights.items()}
        # tree_unflatten reconstructs the nested dict structure nn.Module.update expects
        self.online.update(tree_unflatten(list(weights_mx.items())))
        mx.eval(self.online.parameters())

    def save_weights(self, path: str):
        # Delegates to MLX's built-in .npz serialiser
        self.online.save_weights(path)

    def load_weights(self, path: str):
        self.online.load_weights(path)
        mx.eval(self.online.parameters())
        # Sync so the target net reflects the loaded checkpoint immediately
        self._sync_hard()
