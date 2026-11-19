"""Checkpoint serialisation for CoDARec.

Checkpoints are plain ``state_dict`` payloads: they never pickle a live
``nn.Module``, so they stay loadable across PyTorch versions and devices.  The
resolved :class:`~codarec.config.Config` travels with the weights, which makes
every run reproducible from a single file.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from codarec.config import Config
from codarec.models.denoising_mlp import DenoisingMLP

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str,
    model: nn.Module,
    cfg: Config,
    n_items: int,
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
    optimizer: Optional[optim.Optimizer] = None,
) -> str:
    """Write a checkpoint to ``path`` (parent directories are created)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "n_items": n_items,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, path)
    cfg.save_json(os.path.splitext(path)[0] + "_config.json")
    return path


def read_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Read and lightly validate a CoDARec checkpoint."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            f"{path} is not a CoDARec checkpoint (expected a dict with "
            "'model_state_dict'); legacy pickled modules are not supported"
        )
    return payload


def build_model(cfg: Config, n_items: int, device: torch.device | str) -> DenoisingMLP:
    """Instantiate the denoising network described by ``cfg``."""
    out_dims = list(cfg.dims) + [n_items]
    in_dims = out_dims[::-1]
    return DenoisingMLP(
        in_dims,
        out_dims,
        cfg.emb_size,
        time_type=cfg.time_type,
        norm=cfg.norm,
        dropout=cfg.dropout,
    ).to(device)


def load_model(
    path: str,
    device: torch.device | str = "cpu",
) -> Tuple[DenoisingMLP, Config, Dict[str, Any]]:
    """Restore a trained model, its config and the checkpoint metadata."""
    payload = read_checkpoint(path, map_location=device)
    cfg = Config.from_dict(payload["config"])
    n_items = int(payload["n_items"])
    model = build_model(cfg, n_items, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg, payload


__all__ = [
    "CHECKPOINT_VERSION",
    "build_model",
    "load_model",
    "read_checkpoint",
    "save_checkpoint",
]
