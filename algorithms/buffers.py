import os
import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity circular buffer of (s, a, r, s', done) transitions.

    Random sampling breaks temporal correlations that would otherwise make
    mini-batches non-i.i.d. and destabilise gradient descent.
    When full, the oldest transition is silently dropped (deque maxlen behaviour)
    to make room for the newest, so the buffer always contains recent experience.
    """

    def __init__(self, capacity: int):
        # deque with maxlen automatically evicts the oldest element on overflow
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Cast done to float so it can be multiplied in the Bellman update (1 - done)
        self.buf.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int) -> tuple:
        # random.sample draws without replacement, so no duplicate transitions in a batch
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),                        # preserves stored dtype (uint8 or float32)
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states),                   # preserves stored dtype (uint8 or float32)
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


class _SumTree:
    """
    Binary tree where each leaf holds a priority and each internal node
    holds the sum of its subtree. Supports O(log n) update and sampling.

    The tree is stored as a flat array of size (2*capacity - 1):
      - Indices 0 .. capacity-2  are internal nodes (sums).
      - Indices capacity-1 .. 2*capacity-2  are leaves (one per stored transition).

    Layout (capacity=4):
        index:  0
               / \\
              1   2
             / \\ / \\
            3  4 5  6   ← leaves (data indices 0..3)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # float64 gives enough precision for summing many small priorities
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def _propagate(self, tree_idx: int, delta: float):
        # Walk from the updated leaf up to the root, adding the change at each level
        idx = tree_idx
        while idx > 0:
            idx = (idx - 1) // 2  # parent index in 0-based flat array
            self.tree[idx] += delta

    def _retrieve(self, s: float) -> int:
        """
        Walk from root to leaf, following the cumulative sum s.

        At each node, go left if s fits in the left subtree's total,
        otherwise subtract the left total and descend right. This implements
        inverse-CDF sampling over the leaf priorities in O(log n).
        """
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                # Reached a leaf node — return its index
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    @property
    def total(self) -> float:
        # Root node always holds the sum of all leaf priorities
        return float(self.tree[0])

    def update(self, tree_idx: int, priority: float):
        # Compute change and propagate upward so all ancestor sums stay correct
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, s: float) -> tuple[int, float, int]:
        """
        Return (tree_idx, priority, data_idx) for cumulative sum s.

        data_idx maps the leaf back to the circular data array index.
        Clamping s to (total - epsilon) avoids retrieving a phantom leaf
        when floating-point rounding pushes s slightly past the total.
        """
        s = min(s, self.total - 1e-8)
        tree_idx = self._retrieve(s)
        # Leaves start at index (capacity - 1); subtract to get the data slot
        data_idx = tree_idx - (self.capacity - 1)
        return tree_idx, float(self.tree[tree_idx]), data_idx


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2015).

    Transitions are sampled proportional to |TD error|^alpha, so high-error
    (surprising) transitions are replayed more often. Importance-sampling
    weights correct for the non-uniform sampling bias introduced by
    prioritised replay, preventing the loss from being biased toward
    frequently-replayed transitions.

    Args:
        capacity: Maximum number of transitions to store.
        alpha:    Priority exponent. 0 = uniform sampling, 1 = fully prioritised.
        beta:     IS-weight exponent. 0 = no correction, 1 = full correction.
                  Typically annealed from 0.4 → 1.0 over training.
        epsilon:  Small constant added to |TD error| before raising to alpha,
                  ensuring every transition has non-zero sampling probability.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self._tree = _SumTree(capacity)
        self._data: list = [None] * capacity  # circular data store, same size as tree leaves
        self._write = 0       # next slot to overwrite (circular pointer)
        self._size = 0        # number of valid transitions currently stored
        self._max_priority = 1.0  # tracks running maximum for new-transition priority

    def push(self, state, action, reward, next_state, done):
        self._data[self._write] = (state, action, reward, next_state, float(done))
        # New transitions always get max priority so they are sampled at least once
        # before their real TD error is known. This prevents newly added experience
        # from being starved by high-priority old transitions.
        self._tree.update(
            self._write + self.capacity - 1,  # convert data slot to leaf tree index
            self._max_priority ** self.alpha,
        )
        self._write = (self._write + 1) % self.capacity  # advance circular pointer
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """
        Stratified sampling: divide [0, total) into batch_size equal segments
        and draw one uniform sample from each. This gives better coverage than
        pure uniform sampling over the priority distribution.

        Returns:
            batch:        (states, actions, rewards, next_states, dones)
            tree_indices: indices into the sum-tree (needed for priority update)
            is_weights:   importance-sampling weights, shape (batch_size,)
        """
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities   = np.empty(batch_size, dtype=np.float64)
        data_indices = np.empty(batch_size, dtype=np.int64)

        # Divide the priority range into equal segments for stratified sampling
        segment = self._tree.total / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            ti, pri, di = self._tree.get(s)
            tree_indices[i] = ti
            priorities[i]   = pri
            data_indices[i] = di

        # Importance-sampling weights: w_i = (N * P(i))^{-beta} / max_w
        # Dividing by max_w scales weights into [0, 1] — this is equivalent to
        # normalising so the largest weight is 1, keeping the effective learning
        # rate bounded even when beta < 1.
        probs      = priorities / self._tree.total
        is_weights = (self._size * probs) ** (-self.beta)
        is_weights /= is_weights.max()  # normalise to [0, 1]

        batch = [self._data[i] for i in data_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            (
                np.array(states),                        # preserves stored dtype (uint8 or float32)
                np.array(actions,     dtype=np.int32),
                np.array(rewards,     dtype=np.float32),
                np.array(next_states),                   # preserves stored dtype (uint8 or float32)
                np.array(dones,       dtype=np.float32),
            ),
            tree_indices,
            np.array(is_weights, dtype=np.float32),
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        # Recompute priority = (|td_error| + epsilon)^alpha and push into the tree.
        # epsilon prevents zero-priority (which would make a transition unreachable).
        # _max_priority tracks the raw (pre-alpha) maximum so that push can correctly
        # compute max_raw^alpha for new transitions. Storing alpha-raised values here
        # and then raising again in push would double-apply alpha.
        raw = np.abs(td_errors) + self.epsilon
        priorities = raw ** self.alpha
        for idx, priority, r in zip(tree_indices, priorities, raw):
            self._tree.update(int(idx), float(priority))
            # Track raw max so push uses max_raw^alpha (not max_raw^alpha^alpha)
            self._max_priority = max(self._max_priority, float(r))

    def save(self, ckpt_dir: str) -> None:
        """
        Save buffer state to <ckpt_dir>/replay_buffer.npz atomically.

        Writes to replay_buffer_tmp.npz first, then renames so a crash
        mid-save never corrupts the previous good file. Valid data is always
        in slots 0.._size-1 (the circular write pointer is saved separately
        so the exact slot layout is restored on load).
        """
        final = os.path.join(ckpt_dir, "replay_buffer.npz")
        tmp   = os.path.join(ckpt_dir, "replay_buffer_tmp.npz")

        n = self._size
        items = self._data[:n]
        states      = np.array([x[0] for x in items])
        actions     = np.array([x[1] for x in items], dtype=np.int32)
        rewards     = np.array([x[2] for x in items], dtype=np.float32)
        next_states = np.array([x[3] for x in items])
        dones       = np.array([x[4] for x in items], dtype=np.float32)

        np.savez(
            tmp,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            tree=self._tree.tree,
            write=np.array(self._write),
            size=np.array(n),
            max_priority=np.array(self._max_priority),
        )
        os.replace(tmp, final)

    def load(self, ckpt_dir: str) -> bool:
        """
        Restore buffer state from <ckpt_dir>/replay_buffer.npz.
        Returns True on success, False if no file or load failed.
        """
        path = os.path.join(ckpt_dir, "replay_buffer.npz")
        if not os.path.exists(path):
            return False
        try:
            f = np.load(path, allow_pickle=False)
            n           = int(f["size"])
            states      = f["states"]
            actions     = f["actions"]
            rewards     = f["rewards"]
            next_states = f["next_states"]
            dones       = f["dones"]

            for i in range(n):
                self._data[i] = (
                    states[i],
                    int(actions[i]),
                    float(rewards[i]),
                    next_states[i],
                    float(dones[i]),
                )

            self._tree.tree[:]  = f["tree"]
            self._write         = int(f["write"])
            self._size          = n
            self._max_priority  = float(f["max_priority"])
            return True
        except Exception as e:
            print(f"Warning: failed to load replay buffer ({e}). Starting with empty buffer.")
            return False

    def __len__(self) -> int:
        return self._size
