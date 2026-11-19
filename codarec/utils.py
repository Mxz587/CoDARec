"""Shared utilities: seeding, device resolution and logging."""

from __future__ import annotations

import os
import random
import time
from typing import Callable, Optional, TextIO

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random number generator used by CoDARec."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Return a DataLoader ``worker_init_fn`` that keeps workers reproducible."""

    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(seed + worker_id)

    return worker_init_fn


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def resolve_device(name: str = "auto") -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda`` into a ``torch.device``."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available on this machine; "
            "pass --device cpu instead."
        )
    return torch.device(name)


def count_parameters(module: torch.nn.Module) -> int:
    """Number of trainable parameters in ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class Logger:
    """Minimal tee logger: writes to stdout and to a text file."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._handle: Optional[TextIO] = None
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._handle = open(path, "w", encoding="utf-8")

    def _emit(self, text: str) -> None:
        print(text, flush=True)
        if self._handle is not None:
            self._handle.write(text + "\n")
            self._handle.flush()

    def info(self, *parts: object) -> None:
        self._emit(" ".join(str(part) for part in parts))

    def rule(self, char: str = "-", width: int = 72) -> None:
        self._emit(char * width)

    def timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def ensure_dir(path: str) -> str:
    """Create ``path`` if needed and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def format_seconds(seconds: float) -> str:
    """``3725.4`` -> ``"1h 02m 05s"``."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


__all__ = [
    "Logger",
    "count_parameters",
    "ensure_dir",
    "format_seconds",
    "make_worker_init_fn",
    "resolve_device",
    "set_seed",
]
