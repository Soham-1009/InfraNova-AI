from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.checkpoint import save_checkpoint


class ModelCheckpoint:
    """
    Saves latest and best checkpoints based on validation SSIM.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        monitor: str = "val_ssim",
        mode: str = "max",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")

        (self.checkpoint_dir / "best").mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "latest").mkdir(parents=True, exist_ok=True)

    def _is_improved(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def step(
        self,
        model,
        optimizers: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        scaler: Optional[Any] = None,
    ) -> None:
        latest_path = self.checkpoint_dir / "latest" / "pix2pix_latest.pth"
        save_checkpoint(
            model=model,
            optimizer=optimizers,
            epoch=epoch,
            metrics=metrics,
            path=str(latest_path),
            scaler=scaler,
        )

        score = float(metrics.get(self.monitor, 0.0))
        if self._is_improved(score):
            self.best_score = score
            best_path = self.checkpoint_dir / "best" / "pix2pix_best.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizers,
                epoch=epoch,
                metrics=metrics,
                path=str(best_path),
                scaler=scaler,
            )


class EarlyStopping:
    """
    Early stops training when validation SSIM does not improve.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_ssim",
        mode: str = "max",
    ) -> None:
        self.patience = int(patience)
        self.monitor = monitor
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.stop = False

    def _is_improved(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def step(self, metrics: Dict[str, float]) -> bool:
        score = float(metrics.get(self.monitor, 0.0))

        if self._is_improved(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        return self.stop