from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scaler: Any = None,
) -> None:
    """
    Save a full training checkpoint.

    Args:
        model: model to save
        optimizer: single optimizer or dict of optimizers
        epoch: current epoch
        metrics: metric dictionary
        path: file path
        scaler: optional GradScaler
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }

    if isinstance(optimizer, dict):
        checkpoint["optimizer_state_dict"] = {
            name: opt.state_dict() for name, opt in optimizer.items()
        }
    else:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    scaler: Any = None,
) -> Tuple[int, Dict[str, float]]:
    """
    Load a full training checkpoint.

    Args:
        path: checkpoint file path
        model: model instance
        optimizer: single optimizer or dict of optimizers
        scaler: optional GradScaler

    Returns:
        (epoch, metrics)
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if isinstance(optimizer, dict):
        opt_state = checkpoint.get("optimizer_state_dict", {})
        for name, opt in optimizer.items():
            if name in opt_state:
                opt.load_state_dict(opt_state[name])
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics", {})

    return epoch, metrics