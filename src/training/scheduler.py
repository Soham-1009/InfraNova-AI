from __future__ import annotations

from typing import List


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

        for base_lr, param_group in zip(self.initial_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * lr_mult

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]