import json
import os
import sys

from algorithms.base import BaseAlgorithm


class _TeeStream:
    """Mirror writes to the terminal stream and an append-only log file."""

    def __init__(self, stream, log_path: str):
        self._base_stream = getattr(stream, "_base_stream", stream)
        self._base_log_path = log_path
        self._log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    def write(self, data):
        written = self._base_stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()
        return written

    def flush(self):
        self._base_stream.flush()
        self._log_file.flush()

    def close(self):
        self._log_file.close()

    def __getattr__(self, name):
        return getattr(self._base_stream, name)


class Checkpointer:
    """
    Keeps exactly two checkpoints on disk:
        <ckpt_dir>/latest.npz / latest.json  — overwritten every episode
        <ckpt_dir>/best.npz  / best.json     — overwritten only when best_score improves

    Keeping 'latest' separate from 'best' means training can always resume
    from the most recent episode while 'best' is preserved for evaluation
    without the two interfering with each other.

    Each checkpoint is a pair of files:
      - .npz  — network weights (written by the algorithm's save_weights)
      - .json — metadata (episode number, total steps, best score, epsilon)
    """

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir

    @property
    def log_path(self) -> str:
        """Return the terminal-mirroring log file path for this checkpoint dir."""
        return os.path.join(self.ckpt_dir, "train.log")

    def _paths(self, tag: str) -> tuple[str, str]:
        """Return the (.npz, .json) paths for a given tag ('latest' or 'best')."""
        base = os.path.join(self.ckpt_dir, tag)
        return base + ".npz", base + ".json"

    def install_process_logger(self):
        """
        Mirror stdout/stderr into <ckpt_dir>/train.log for the current process.

        This is intentionally lightweight: training code keeps using print(),
        while the tee stream preserves the same terminal output in the log file.
        """
        os.makedirs(self.ckpt_dir, exist_ok=True)
        install_process_logger(self.log_path)

    def save(self, algo: BaseAlgorithm, meta: dict, is_best: bool = False, is_best_score: bool = False):
        """
        Write the latest checkpoint and, if flagged, overwrite the best checkpoints too.

        makedirs with exist_ok=True means the checkpoint directory is created
        on the first save and subsequent saves are no-ops for the directory.
        Both the weights and the metadata are always written atomically
        (file-at-a-time) so a crash mid-save can only corrupt the file
        being written, not the other one.
        """
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Always update the latest checkpoint so training can resume
        weights_path, meta_path = self._paths("latest")
        algo.save_weights(weights_path)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Overwrite best_avg100 checkpoint when rolling average improves
        if is_best:
            weights_path, meta_path = self._paths("best")
            algo.save_weights(weights_path)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        # Overwrite best_score checkpoint when peak episode score improves
        if is_best_score:
            weights_path, meta_path = self._paths("best_score")
            algo.save_weights(weights_path)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    def load(self, algo: BaseAlgorithm) -> dict | None:
        """
        Load the latest checkpoint into algo. Returns meta dict or None.

        Returns None (instead of raising) when no checkpoint exists, so
        callers can treat a fresh run and a resumed run uniformly.
        """
        weights_path, meta_path = self._paths("latest")
        if not os.path.exists(weights_path):
            # No checkpoint saved yet — caller should start from scratch
            return None
        try:
            algo.load_weights(weights_path)
            with open(meta_path) as f:
                return json.load(f)
        except Exception as e:
            # Corrupt or incompatible checkpoint — warn and fall back to fresh start
            print(f"Warning: failed to load checkpoint ({e}). Starting fresh.")
            return None

    def load_best_score(self, algo: BaseAlgorithm) -> dict | None:
        """Load the best peak-score checkpoint into algo. Returns meta dict or None."""
        weights_path, meta_path = self._paths("best_score")
        if not os.path.exists(weights_path):
            return None
        try:
            algo.load_weights(weights_path)
            with open(meta_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: failed to load best_score checkpoint ({e}).")
            return None

    def load_best(self, algo: BaseAlgorithm) -> dict | None:
        """
        Load the best checkpoint into algo. Returns meta dict or None.

        Used during evaluation (--test --best) to restore the highest-scoring
        policy seen during training rather than the most recently saved one.
        """
        weights_path, meta_path = self._paths("best")
        if not os.path.exists(weights_path):
            return None
        try:
            algo.load_weights(weights_path)
            with open(meta_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: failed to load best checkpoint ({e}). Starting fresh.")
            return None


def install_process_logger(log_path: str):
    """Mirror stdout/stderr into the provided log file for the current process."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if getattr(stream, "_base_log_path", None) == log_path:
            continue

        if isinstance(stream, _TeeStream):
            stream.close()

        setattr(sys, stream_name, _TeeStream(stream, log_path))
