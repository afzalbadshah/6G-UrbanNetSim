import os
import random
from collections import deque

import numpy as np
import torch


def _configure_torch_threads():
    num_threads = int(os.environ.get("SMARTCITY_TORCH_THREADS", "1") or 1)
    interop_threads = int(os.environ.get("SMARTCITY_TORCH_INTEROP_THREADS", "1") or 1)
    try:
        torch.set_num_threads(max(1, num_threads))
        torch.set_num_interop_threads(max(1, interop_threads))
    except RuntimeError:
        # PyTorch only allows interop threads to be set before parallel work starts.
        pass


_configure_torch_threads()


class NoFeasibleActionError(RuntimeError):
    """Raised when a policy is called with an explicit all-zero mask."""


def set_global_seed(seed):
    """Seed every RNG stream and request deterministic kernels (Task 57).

    Seeds Python ``random``, NumPy, and Torch (CPU + all CUDA devices), and asks
    Torch/cuDNN for deterministic algorithms so a fixed seed reproduces a run.
    ``warn_only=True`` keeps a missing deterministic kernel from crashing the run
    (it warns instead), and ``PYTHONHASHSEED`` is pinned for hash-order stability.
    All calls are best-effort so seeding never fails on a CPU-only or older build.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        # Required by some CUDA GEMM kernels before deterministic mode is honored.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def valid_actions_from_mask(mask, action_dim=None):
    if mask is None:
        return list(range(int(action_dim or 0)))
    limit = int(action_dim or len(mask))
    return [
        idx
        for idx, value in enumerate(mask[:limit])
        if int(value) == 1
    ]


def masked_random_action(mask, action_dim):
    valid = valid_actions_from_mask(mask, action_dim)
    if not valid:
        raise NoFeasibleActionError("no physically feasible action")
    return int(random.choice(valid))


def masked_argmax(values, mask=None):
    arr = np.asarray(values, dtype=np.float32).copy()
    if mask is not None:
        valid = valid_actions_from_mask(mask, arr.size)
        if not valid:
            raise NoFeasibleActionError("no physically feasible action")
        for idx, value in enumerate(mask):
            if idx < arr.size and int(value) != 1:
                arr[idx] = -1e9
    best = np.max(arr)
    candidates = np.flatnonzero(np.isclose(arr, best))
    if candidates.size == 0:
        return int(np.argmax(arr))
    return int(np.random.choice(candidates))


def to_tensor(value, dtype=torch.float32):
    return torch.as_tensor(np.asarray(value), dtype=dtype)


def save_checkpoint(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


class EpsilonScheduler:
    def __init__(self, start=1.0, minimum=0.05, decay=0.995, exploration_steps=0, decay_steps=0):
        self.start = float(np.clip(start, 0.0, 1.0))
        self.minimum = min(float(np.clip(minimum, 0.0, 1.0)), self.start)
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.exploration_steps = int(max(0, exploration_steps))
        self.decay_steps = int(max(0, decay_steps))
        self.value = self.start
        self.step_count = 0

    def reset(self, training):
        self.value = self.start if bool(training) else 0.0
        self.step_count = 0
        return self.value

    def value_at(self, step_count):
        step = int(max(0, step_count))
        if step < self.exploration_steps:
            return self.start
        decay_step = max(0, step - self.exploration_steps)
        if self.decay_steps > 0:
            progress = min(1.0, decay_step / float(self.decay_steps))
            return self.start + (self.minimum - self.start) * progress
        return max(self.minimum, self.start * (self.decay ** decay_step))

    def set_step(self, step_count):
        self.step_count = int(max(0, step_count))
        self.value = float(np.clip(self.value_at(self.step_count), self.minimum, self.start))
        return self.value

    def step(self):
        return self.set_step(self.step_count + 1)


class ReplayBuffer:
    def __init__(self, capacity):
        self._items = deque(maxlen=int(capacity))

    def append(self, item):
        self._items.append(dict(item))

    def __len__(self):
        return len(self._items)

    def sample(self, batch_size):
        indices = np.random.choice(len(self._items), int(batch_size), replace=False)
        return [self._items[int(i)] for i in indices]
