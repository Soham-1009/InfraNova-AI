from __future__ import annotations

import logging
import shutil
from datetime import UTC
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.datasets.landsat9_dataset import Landsat9Dataset
from src.models.pix2pix.pix2pix import Pix2Pix
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config_validator import validate_config
from src.utils.seed import seed_everything

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
    def _normalize(sample: Any) -> dict[str, torch.Tensor]:
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._normalize(self.base_dataset[idx])


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]

    root_dir = dataset_cfg["root_dir"]
    image_size = int(dataset_cfg.get("image_size", 256))
    num_workers = int(dataset_cfg.get("num_workers", 2))
    batch_size = int(training_cfg.get("batch_size", 8))
    subset_ratio = dataset_cfg.get("subset_ratio", None)
    subset_seed = int(dataset_cfg.get("subset_seed", 42))

    # Normalization config (backward-compatible: defaults to "local")
    norm_cfg = dataset_cfg.get("normalization", {})
    normalization = str(norm_cfg.get("mode", "local"))
    stats_file = norm_cfg.get("stats_file", None)

    train_base = Landsat9Dataset(
        root_dir=root_dir,
        split="train",
        image_size=image_size,
        normalization=normalization,
        stats_file=stats_file,
        subset_ratio=subset_ratio,
        subset_seed=subset_seed,
    )
    val_base = Landsat9Dataset(
        root_dir=root_dir,
        split="val",
        image_size=image_size,
        augment=False,
        normalization=normalization,
        stats_file=stats_file,
        subset_ratio=subset_ratio,
        subset_seed=subset_seed,
    )

    train_dataset = LandsatBatchAdapter(train_base)
    val_dataset = LandsatBatchAdapter(val_base)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return train_loader, val_loader


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _save_experiment_json(
    cfg: dict[str, Any],
    dataset_info: dict[str, Any],
    best_ssim: float,
    best_psnr: float,
    checkpoint_dir: Path
) -> None:
    """Save an experiment.json and config.yaml alongside the checkpoint."""
    import json
    import platform
    import hashlib
    import yaml
    
    experiment_id = cfg.get("project", {}).get("name", "InfraNova-AI").replace(" ", "_")
    
    # Extract manifest if possible
    manifest_hash = "unknown"
    manifest_path = Path(cfg.get("dataset", {}).get("root_dir", "")).parent / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "rb") as f:
            manifest_hash = hashlib.sha256(f.read()).hexdigest()[:12]
            
    # Write config copy
    config_copy_path = checkpoint_dir / f"{experiment_id}_config.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
        
    with open(config_copy_path, "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        
    exp_data = {
        "experiment_id": experiment_id,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "None",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "config_hash": config_hash,
        "dataset_fingerprint": manifest_hash,
        "train_samples": dataset_info.get("train_samples", 0),
        "val_samples": dataset_info.get("val_samples", 0),
        "subset_ratio": cfg.get("dataset", {}).get("subset_ratio", None),
        "subset_seed": cfg.get("dataset", {}).get("subset_seed", 42),
        "best_ssim": float(best_ssim),
        "best_psnr": float(best_psnr)
    }
    
    exp_path = checkpoint_dir / f"{experiment_id}_experiment.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(exp_data, f, indent=2)

def _save_experiment_comparison(cfg: dict[str, Any], history: dict[str, list]) -> None:
    """Append a row to logs/experiment_comparison.csv with config and best metrics."""
    import csv
    from datetime import datetime

    training_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", training_cfg.get("loss", {}))
    dataset_cfg = cfg.get("dataset", {})
    norm_cfg = dataset_cfg.get("normalization", {})
    sched_cfg = cfg.get("scheduler", training_cfg.get("scheduler", {}))

    # Find best metrics from history
    val_ssim_values = history.get("val_ssim", [])
    val_psnr_values = history.get("val_psnr", [])

    best_ssim = max(val_ssim_values) if val_ssim_values else 0.0
    best_psnr = max(val_psnr_values) if val_psnr_values else 0.0

    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment_name": cfg.get("project", {}).get("name", "InfraNova AI"),
        "normalization_mode": norm_cfg.get("mode", "local"),
        "lambda_adv": loss_cfg.get("lambda_adv", 1.0),
        "lambda_l1": loss_cfg.get("lambda_l1", 10.0),
        "lambda_perc": loss_cfg.get("lambda_perc", 10.0),
        "lambda_ssim": loss_cfg.get("lambda_ssim", 5.0),
        "lambda_chroma": loss_cfg.get("lambda_chroma", 0.0),
        "lambda_feat": loss_cfg.get("lambda_feat", 0.0),
        "gan_mode": loss_cfg.get("gan_mode", "bce"),
        "scheduler_type": sched_cfg.get("type", "linear"),
        "multi_scale_disc": cfg.get("model", {}).get("multi_scale_disc", False),
        "epochs_trained": len(val_ssim_values),
        "best_ssim": f"{best_ssim:.6f}",
        "best_psnr": f"{best_psnr:.4f}",
        "lr": training_cfg.get("optimizer", {}).get("lr", 0.0002),
        "batch_size": training_cfg.get("batch_size", 8),
    }

    csv_path = Path(cfg.get("paths", {}).get("logs", "logs")) / "experiment_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info("Experiment comparison row appended to %s", csv_path)


def run_training(cfg: dict[str, Any]) -> dict[str, list]:
    """
    Train Pix2Pix on Landsat 9 TIR->RGB data using the existing Trainer class.

    This function keeps the existing Trainer unchanged and controls:
        - LR schedule
        - checkpoint resumption
        - early stopping
        - best/final checkpoint saving
    """
    seed_everything(int(cfg.get("project", {}).get("seed", 42)))

    # Validate configuration before any expensive work
    validate_config(cfg)

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

    # Multi-scale discriminator support
    multi_scale = bool(cfg.get("model", {}).get("multi_scale_disc", False))

    model = Pix2Pix(
        device=device,
        in_channels=int(dataset_cfg.get("input_channels", 1)),
        out_channels=int(dataset_cfg.get("output_channels", 3)),
        image_size=int(dataset_cfg.get("image_size", 256)),
        multi_scale=multi_scale,
        generator_impl=cfg.get("model", {}).get("generator", {}).get("implementation", "legacy"),
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
    _set_optimizer_lr(trainer.optimizer_d, base_lr * 0.5)

    # Build scheduler from config — top-level key, fallback to training.scheduler
    sched_cfg = cfg.get("scheduler", training_cfg.get("scheduler", {}))
    sched_type = str(sched_cfg.get("type", "linear"))

    scheduler_g = build_scheduler(
        scheduler_type=sched_type,
        optimizer=trainer.optimizer_g,
        total_epochs=epochs,
        decay_start_epoch=decay_start_epoch,
        T_0=int(sched_cfg.get("T_0", 50)),
        T_mult=int(sched_cfg.get("T_mult", 2)),
        eta_min=float(sched_cfg.get("eta_min", 1e-6)),
    )
    scheduler_d = build_scheduler(
        scheduler_type=sched_type,
        optimizer=trainer.optimizer_d,
        total_epochs=epochs,
        decay_start_epoch=decay_start_epoch,
        T_0=int(sched_cfg.get("T_0", 50)),
        T_mult=int(sched_cfg.get("T_mult", 2)),
        eta_min=float(sched_cfg.get("eta_min", 1e-6)),
    )

    history: dict[str, list] = {
        "g_loss": [],
        "d_loss": [],
        "l1": [],
        "adv": [],
        "perc": [],
        "ssim": [],
        "chroma": [],
        "feat": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    logger.info(
        "Starting training: device=%s, batch_size=%d, epochs=%d, scheduler=%s",
        device,
        batch_size,
        epochs,
        sched_type,
    )

    # Log experiment info at start
    trainer.logger.log_experiment_info(config=cfg, phase="start")

    dataset_info = {
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
    }

    for epoch in range(start_epoch, epochs):
        scheduler_g.step(epoch)
        scheduler_d.step(epoch)

        train_metrics = trainer.train_one_epoch(epoch)
        val_metrics = trainer.validate()

        epoch_metrics = {**train_metrics, **val_metrics}

        trainer.logger.log_epoch(epoch + 1, epoch_metrics)

        for k in history:
            history[k].append(float(epoch_metrics.get(k, 0.0)))

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
        _save_experiment_json(
            cfg,
            dataset_info,
            best_val_ssim,
            val_metrics["val_psnr"],
            Path(latest_path).parent
        )

        # Save epoch-specific checkpoint for resume testing
        epoch_path = Path(paths_cfg["checkpoints"]) / f"epoch_{epoch + 1}.pth"
        save_checkpoint(
            model=trainer.model,
            optimizer={
                "generator": trainer.optimizer_g,
                "discriminator": trainer.optimizer_d,
            },
            epoch=epoch + 1,
            metrics=epoch_metrics,
            path=str(epoch_path),
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
            _save_experiment_json(
                cfg,
                dataset_info,
                best_val_ssim,
                val_metrics["val_psnr"],
                Path(best_path).parent
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

    # Log experiment info at end
    trainer.logger.log_experiment_info(phase="end")

    # Save experiment comparison row
    _save_experiment_comparison(cfg, history)

    # Final checkpoint copy
    final_path = paths_cfg["final_checkpoint"]
    if Path(paths_cfg["latest_checkpoint"]).exists():
        shutil.copy2(paths_cfg["latest_checkpoint"], final_path)
        logger.info("Saved final checkpoint to %s", final_path)
        
        # Copy the latest experiment config and json to final directory
        latest_dir = Path(paths_cfg["latest_checkpoint"]).parent
        final_dir = Path(final_path).parent
        if latest_dir != final_dir:
            for json_file in latest_dir.glob("*_experiment.json"):
                shutil.copy2(json_file, final_dir / json_file.name)
            for yaml_file in latest_dir.glob("*_config.yaml"):
                shutil.copy2(yaml_file, final_dir / yaml_file.name)

    return history


def main() -> None:
    cfg = load_config("configs/config.yaml")
    run_training(cfg)


if __name__ == "__main__":
    main()

