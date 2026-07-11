from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


class TrainingLogger:
    """
    CSV training logger with optional Weights & Biases integration.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        use_wandb: bool = False,
        project_name: str = "InfraNova-AI",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "training.csv"
        self.plot_path = self.log_dir / "loss_curve.png"
        self.use_wandb = use_wandb

        self.rows: List[Dict[str, Any]] = []

        self._wandb = None
        if self.use_wandb:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(project=project_name, name="pix2pix-training", reinit=True)
            except Exception:
                self._wandb = None
                self.use_wandb = False

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        row = {"epoch": epoch, **metrics}
        self.rows.append(row)

        self._write_csv()
        if self._wandb is not None:
            self._wandb.log(row)

    def _write_csv(self) -> None:
        if not self.rows:
            return

        fieldnames = list(dict.fromkeys(key for row in self.rows for key in row))
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def save_plot(self) -> None:
        if not self.rows:
            return

        epochs = [row["epoch"] for row in self.rows]

        keys = [
            "g_loss",
            "d_loss",
            "l1",
            "adv",
            "perc",
            "ssim",
            "val_psnr",
            "val_ssim",
        ]

        plt.figure(figsize=(12, 8))

        for key in keys:
            if key in self.rows[0]:
                values = [row[key] for row in self.rows]
                plt.plot(epochs, values, label=key)

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300, bbox_inches="tight")
        plt.close()