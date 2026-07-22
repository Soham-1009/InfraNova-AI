from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch


class TrainingLogger:
    """
    CSV training logger with optional Weights & Biases and TensorBoard integration,
    experiment tracking, and individual training curve PNGs.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        project_name: str = "InfraNova-AI",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "training.csv"
        self.plot_path = self.log_dir / "loss_curve.png"
        self.experiment_path = self.log_dir / "experiment_info.json"
        self.experiment_csv_path = self.log_dir / "experiment_history.csv"
        self.use_wandb = use_wandb

        self.rows: List[Dict[str, Any]] = []

        # Weights & Biases
        self._wandb = None
        if self.use_wandb:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(project=project_name, name="pix2pix-training", reinit=True)
            except Exception:
                self._wandb = None
                self.use_wandb = False

        # TensorBoard
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            except ImportError:
                pass  # TensorBoard not available
            except Exception:
                pass

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        row = {"epoch": epoch, **metrics}
        self.rows.append(row)

        self._write_csv()

        if self._wandb is not None:
            self._wandb.log(row)

        if self._tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, epoch)
            self._tb_writer.flush()

    def _write_csv(self) -> None:
        if not self.rows:
            return

        fieldnames = list(dict.fromkeys(key for row in self.rows for key in row))
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def log_experiment_info(
        self,
        config: Optional[Dict[str, Any]] = None,
        phase: str = "start",
    ) -> None:
        """
        Save experiment metadata to JSON and CSV.

        Args:
            config: Training configuration dictionary.
            phase: "start" or "end".
        """
        # Load existing JSON info if updating
        info: Dict[str, Any] = {}
        if self.experiment_path.exists():
            try:
                info = json.loads(self.experiment_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        if phase == "start":
            info["timestamp_start"] = datetime.now(timezone.utc).isoformat()
            info["python_version"] = sys.version
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
            info["device"] = "cuda" if torch.cuda.is_available() else "cpu"

            # Git commit
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    info["git_commit"] = result.stdout.strip()
            except Exception:
                info["git_commit"] = "unavailable"

            if config is not None:
                info["hyperparameters"] = config

        if phase == "end":
            info["timestamp_end"] = datetime.now(timezone.utc).isoformat()

            if self.rows:
                val_ssim_values = [
                    (r.get("epoch", 0), r.get("val_ssim", 0.0))
                    for r in self.rows if "val_ssim" in r
                ]
                val_psnr_values = [
                    (r.get("epoch", 0), r.get("val_psnr", 0.0))
                    for r in self.rows if "val_psnr" in r
                ]

                if val_ssim_values:
                    best_ssim_epoch, best_ssim = max(val_ssim_values, key=lambda x: x[1])
                    info["best_ssim"] = best_ssim
                    info["best_ssim_epoch"] = best_ssim_epoch

                if val_psnr_values:
                    best_psnr_epoch, best_psnr = max(val_psnr_values, key=lambda x: x[1])
                    info["best_psnr"] = best_psnr
                    info["best_psnr_epoch"] = best_psnr_epoch

                info["total_epochs_trained"] = len(self.rows)

            # Append to experiment history CSV
            self._append_experiment_csv(info)

        # Write JSON
        self.experiment_path.write_text(
            json.dumps(info, indent=2, default=str),
            encoding="utf-8",
        )

    def _append_experiment_csv(self, info: Dict[str, Any]) -> None:
        """Append a summary row to the experiment history CSV."""
        row = {
            "timestamp": info.get("timestamp_end", ""),
            "epochs": info.get("total_epochs_trained", 0),
            "best_ssim": info.get("best_ssim", ""),
            "best_ssim_epoch": info.get("best_ssim_epoch", ""),
            "best_psnr": info.get("best_psnr", ""),
            "best_psnr_epoch": info.get("best_psnr_epoch", ""),
            "device": info.get("device", ""),
            "git_commit": info.get("git_commit", ""),
        }

        file_exists = self.experiment_csv_path.exists()
        with open(self.experiment_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def save_plot(self) -> None:
        """Save combined training curves and individual metric PNGs."""
        if not self.rows:
            return

        epochs = [row["epoch"] for row in self.rows]
        all_keys = set(key for row in self.rows for key in row)

        # Combined plot
        combined_keys = [
            "g_loss", "d_loss", "l1", "adv", "perc", "ssim",
            "val_psnr", "val_ssim", "grad_norm_g", "grad_norm_d", "lr",
        ]

        plt.figure(figsize=(12, 8))
        for key in combined_keys:
            if key not in all_keys:
                continue
            values = [row.get(key) for row in self.rows]
            plot_epochs = [e for e, v in zip(epochs, values) if v is not None]
            plot_values = [v for v in values if v is not None]
            if plot_values:
                plt.plot(plot_epochs, plot_values, label=key)

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Individual plots
        individual_plots = {
            "loss.png": {
                "keys": ["g_loss", "d_loss"],
                "title": "Generator & Discriminator Loss",
                "ylabel": "Loss",
            },
            "ssim.png": {
                "keys": ["ssim", "val_ssim"],
                "title": "SSIM (Train & Validation)",
                "ylabel": "SSIM",
            },
            "psnr.png": {
                "keys": ["val_psnr"],
                "title": "Validation PSNR",
                "ylabel": "PSNR (dB)",
            },
            "lr.png": {
                "keys": ["lr"],
                "title": "Learning Rate Schedule",
                "ylabel": "Learning Rate",
            },
            "grad_norms.png": {
                "keys": ["grad_norm_g", "grad_norm_d"],
                "title": "Gradient Norms",
                "ylabel": "L2 Norm",
            },
        }

        for filename, spec in individual_plots.items():
            available = [k for k in spec["keys"] if k in all_keys]
            if not available:
                continue

            plt.figure(figsize=(10, 5))
            for key in available:
                values = [row.get(key) for row in self.rows]
                plot_epochs = [e for e, v in zip(epochs, values) if v is not None]
                plot_values = [v for v in values if v is not None]
                if plot_values:
                    plt.plot(plot_epochs, plot_values, label=key, linewidth=1.5)

            plt.xlabel("Epoch")
            plt.ylabel(spec["ylabel"])
            plt.title(spec["title"])
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.log_dir / filename, dpi=200, bbox_inches="tight")
            plt.close()

    def close(self) -> None:
        """Clean up resources."""
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass