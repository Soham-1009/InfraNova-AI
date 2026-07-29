from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class LinearLRScheduler:
    """
    Pix2Pix-style learning rate scheduler.

    Keeps LR constant until `decay_start_epoch`, then linearly decays to zero.
    """

    def __init__(
        self,
        optimizer,
        total_epochs: int,
        decay_start_epoch: int,
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.decay_start_epoch = int(decay_start_epoch)
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        if epoch < self.decay_start_epoch:
            lr_mult = 1.0
        else:
            decay_epochs = max(self.total_epochs - self.decay_start_epoch, 1)
            lr_mult = max(0.0, 1.0 - (epoch - self.decay_start_epoch) / decay_epochs)

        for base_lr, param_group in zip(self.initial_lrs, self.optimizer.param_groups, strict=False):
            param_group["lr"] = base_lr * lr_mult

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class CosineScheduler:
    """
    Cosine annealing with warm restarts scheduler.

    Wraps PyTorch's CosineAnnealingWarmRestarts with a consistent
    interface matching LinearLRScheduler.

    Args:
        optimizer: The optimizer to schedule.
        T_0: Number of epochs for the first cosine cycle.
        T_mult: Multiplier for successive cycle lengths.
        eta_min: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer,
        T_0: int = 50,
        T_mult: int = 2,
        eta_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(T_0),
            T_mult=int(T_mult),
            eta_min=float(eta_min),
        )

    def step(self, epoch: int) -> None:
        self.scheduler.step(epoch)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def build_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    total_epochs: int = 250,
    decay_start_epoch: int = 200,
    T_0: int = 50,
    T_mult: int = 2,
    eta_min: float = 1e-6,
):
    """
    Factory function to build the configured scheduler.

    Args:
        scheduler_type: "linear" or "cosine".
        optimizer: The optimizer to schedule.
        total_epochs: Total training epochs (for linear scheduler).
        decay_start_epoch: Epoch to start decay (for linear scheduler).
        T_0: First cycle length (for cosine scheduler).
        T_mult: Cycle length multiplier (for cosine scheduler).
        eta_min: Minimum LR (for cosine scheduler).

    Returns:
        A scheduler instance with .step(epoch) and .get_last_lr() methods.
    """
    stype = scheduler_type.lower().strip()
    if stype == "linear":
        return LinearLRScheduler(
            optimizer=optimizer,
            total_epochs=total_epochs,
            decay_start_epoch=decay_start_epoch,
        )
    elif stype == "cosine":
        return CosineScheduler(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: '{scheduler_type}'. Use 'linear' or 'cosine'."
        )
