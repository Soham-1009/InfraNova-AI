from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.datasets.landsat9_dataset import Landsat9Dataset
from src.models.pix2pix.pix2pix import Pix2Pix
from src.training.losses import CombinedLoss
from src.training.scheduler import LinearLRScheduler
from src.training.trainer import Trainer
from src.utils.checkpoint import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class LandsatBatchAdapter(Dataset):
    """
    Adapter to normalize dataset output keys for the existing Trainer.

    Expected trainer keys:
        batch["ir"], batch["rgb"]

    This adapter accepts common alternatives:
        - {"input", "target"}
        - {"tir", "rgb"}
        - {"ir", "rgb"}
        - tuple/list of length 2
    """

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    @staticmethod
    def _normalize(sample: Any) -> Dict[str, torch.Tensor]:
        if isinstance(sample, dict):
            if "ir" in sample and "rgb" in sample:
                return {"ir": sample["ir"], "rgb": sample["rgb"]}
            if "input" in sample and "target" in sample:
                return {"ir": sample["input"], "rgb": sample["target"]}
            if "tir" in sample and "rgb" in sample:
                return {"ir": sample["tir"], "rgb": sample["rgb"]}
            if "tir" in sample and "target" in sample:
                return {"ir": sample["tir"], "rgb": sample["target"]}
            raise KeyError(
                "Dataset sample must contain one of the following key pairs: "
                "('ir','rgb'), ('input','target'), ('tir','rgb'), ('tir','target')."
            )

        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            return {"ir": sample[0], "rgb": sample[1]}

        raise TypeError(f"Unsupported dataset sample type: {type(sample)}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._normalize(self.base_dataset[idx])


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]

    root_dir = dataset_cfg["root_dir"]
    image_size = int(dataset_cfg.get("image_size", 256))
    num_workers = int(dataset_cfg.get("num_workers", 2))
    batch_size = int(training_cfg.get("batch_size", 8))

    train_base = Landsat9Dataset(
        root_dir=root_dir,
        split="train",
        image_size=image_size,
    )
    val_base = Landsat9Dataset(
        root_dir=root_dir,
        split="val",
        image_size=image_size,
    )

    train_dataset = LandsatBatchAdapter(train_base)
    val_dataset = LandsatBatchAdapter(val_base)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return train_loader, val_loader


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def run_training(cfg: Dict[str, Any]) -> Dict[str, list]:
    """
    Train Pix2Pix on Landsat 9 TIR->RGB data using the existing Trainer class.

    This function keeps the existing Trainer unchanged and controls:
        - LR schedule
        - checkpoint resumption
        - early stopping
        - best/final checkpoint saving
    """
    seed_everything(int(cfg.get("project", {}).get("seed", 42)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    paths_cfg = cfg["paths"]

    batch_size = int(training_cfg.get("batch_size", 8))
    epochs = int(training_cfg.get("epochs", 100))
    decay_start_epoch = int(training_cfg.get("decay_start_epoch", 80))
    patience = int(training_cfg.get("patience", 30))
    resume_from = training_cfg.get("resume_from", "")

    Path(paths_cfg["checkpoints"]).mkdir(parents=True, exist_ok=True)
    Path(paths_cfg["logs"]).mkdir(parents=True, exist_ok=True)
    Path(paths_cfg["outputs"]).mkdir(parents=True, exist_ok=True)
    Path(paths_cfg["visualizations"]).mkdir(parents=True, exist_ok=True)
    Path(Path(paths_cfg["checkpoints"]) / "best").mkdir(parents=True, exist_ok=True)
    Path(Path(paths_cfg["checkpoints"]) / "latest").mkdir(parents=True, exist_ok=True)
    Path(Path(paths_cfg["checkpoints"]) / "final").mkdir(parents=True, exist_ok=True)

    root_dir = Path(dataset_cfg["root_dir"])
    if not root_dir.exists():
        raise FileNotFoundError(f"Landsat 9 dataset directory not found: {root_dir}")

    train_loader, val_loader = build_dataloaders(cfg)

    model = Pix2Pix(
        device=device,
        in_channels=int(dataset_cfg.get("input_channels", 1)),
        out_channels=int(dataset_cfg.get("output_channels", 3)),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
    )

    start_epoch = 0
    best_val_ssim = -float("inf")
    no_improve = 0

    # Resume checkpoint if present
    if resume_from and Path(resume_from).exists():
        logger.info("Resuming from checkpoint: %s", resume_from)
        checkpoint_epoch, checkpoint_metrics = load_checkpoint(
            path=resume_from,
            model=trainer.model,
            optimizer={
                "generator": trainer.optimizer_g,
                "discriminator": trainer.optimizer_d,
            },
            scaler=trainer.scaler,
        )
        start_epoch = int(checkpoint_epoch)

        if isinstance(checkpoint_metrics, dict):
            best_val_ssim = float(checkpoint_metrics.get("val_ssim", best_val_ssim))
            logger.info("Resumed epoch=%d, best_val_ssim=%.4f", start_epoch, best_val_ssim)

    # Reset LR to configured base LR before schedule steps
    base_lr = float(training_cfg["optimizer"]["lr"])
    _set_optimizer_lr(trainer.optimizer_g, base_lr)
    _set_optimizer_lr(trainer.optimizer_d, base_lr)

    scheduler_g = LinearLRScheduler(
        optimizer=trainer.optimizer_g,
        total_epochs=epochs,
        decay_start_epoch=decay_start_epoch,
    )
    scheduler_d = LinearLRScheduler(
        optimizer=trainer.optimizer_d,
        total_epochs=epochs,
        decay_start_epoch=decay_start_epoch,
    )

    history: Dict[str, list] = {
        "g_loss": [],
        "d_loss": [],
        "l1": [],
        "adv": [],
        "perc": [],
        "ssim": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    logger.info(
        "Starting training: device=%s, batch_size=%d, epochs=%d, decay_start=%d",
        device,
        batch_size,
        epochs,
        decay_start_epoch,
    )

    for epoch in range(start_epoch, epochs):
        scheduler_g.step(epoch)
        scheduler_d.step(epoch)

        train_metrics = trainer.train_one_epoch(epoch)
        val_metrics = trainer.validate()

        epoch_metrics = {**train_metrics, **val_metrics}

        trainer.logger.log_epoch(epoch + 1, epoch_metrics)

        for k in history:
            history[k].append(float(epoch_metrics[k]))

        # Latest checkpoint
        latest_path = paths_cfg["latest_checkpoint"]
        save_checkpoint(
            model=trainer.model,
            optimizer={
                "generator": trainer.optimizer_g,
                "discriminator": trainer.optimizer_d,
            },
            epoch=epoch + 1,
            metrics=epoch_metrics,
            path=latest_path,
            scaler=trainer.scaler,
        )

        # Best checkpoint based on validation SSIM
        if val_metrics["val_ssim"] > best_val_ssim:
            best_val_ssim = float(val_metrics["val_ssim"])
            no_improve = 0

            best_path = paths_cfg["best_checkpoint"]
            save_checkpoint(
                model=trainer.model,
                optimizer={
                    "generator": trainer.optimizer_g,
                    "discriminator": trainer.optimizer_d,
                },
                epoch=epoch + 1,
                metrics=epoch_metrics,
                path=best_path,
                scaler=trainer.scaler,
            )
            logger.info("Epoch %d: new best val_ssim=%.4f", epoch + 1, best_val_ssim)
        else:
            no_improve += 1

        # Sample outputs every 5 epochs
        if (epoch + 1) % int(training_cfg.get("sample_every", 5)) == 0:
            trainer._save_sample_images(epoch + 1)  # existing trainer method

        logger.info(
            "Epoch %d/%d | G=%.4f D=%.4f L1=%.4f Adv=%.4f Perc=%.4f SSIM=%.4f | val_PSNR=%.4f val_SSIM=%.4f",
            epoch + 1,
            epochs,
            train_metrics["g_loss"],
            train_metrics["d_loss"],
            train_metrics["l1"],
            train_metrics["adv"],
            train_metrics["perc"],
            train_metrics["ssim"],
            val_metrics["val_psnr"],
            val_metrics["val_ssim"],
        )

        if no_improve >= patience:
            logger.info(
                "Early stopping triggered at epoch %d after %d epochs without val SSIM improvement.",
                epoch + 1,
                patience,
            )
            break

    trainer.logger.save_plot()

    # Final checkpoint copy
    final_path = paths_cfg["final_checkpoint"]
    if Path(paths_cfg["latest_checkpoint"]).exists():
        shutil.copy2(paths_cfg["latest_checkpoint"], final_path)
        logger.info("Saved final checkpoint to %s", final_path)

    return history


def main() -> None:
    cfg = load_config("configs/config.yaml")
    run_training(cfg)


if __name__ == "__main__":
    main()