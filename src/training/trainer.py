from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import CombinedLoss
from src.training.scheduler import LinearLRScheduler
from src.utils.logger import TrainingLogger


class Trainer:
    """Pix2Pix trainer with two-step GAN optimisation."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        training_cfg = config.get("training", {})
        optim_cfg = training_cfg.get("optimizer", {})

        self.lr = float(optim_cfg.get("lr", 2e-4))
        self.beta1 = float(optim_cfg.get("beta1", 0.5))
        self.beta2 = float(optim_cfg.get("beta2", 0.999))
        self.grad_clip = float(training_cfg.get("grad_clip", 1.0))
        self.sample_every = int(training_cfg.get("sample_every", 5))
        self.total_epochs = int(training_cfg.get("epochs", 150))

        self.checkpoint_dir = Path(
            config.get("paths", {}).get("checkpoints", "checkpoints")
        )
        self.visual_dir = (
            Path(config.get("paths", {}).get("outputs", "outputs")) / "visualizations"
        )
        self.log_dir = Path(config.get("paths", {}).get("logs", "logs"))

        self.visual_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generator optimizer - full learning rate
        self.optimizer_g = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

        # Discriminator optimizer - half learning rate to prevent overpowering
        self.optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.lr * 0.5,
            betas=(self.beta1, self.beta2),
        )

        loss_cfg = training_cfg.get("loss", training_cfg)
        self.criterion = CombinedLoss(
            lambda_adv=float(loss_cfg.get("lambda_adv", 0.5)),
            lambda_l1=float(loss_cfg.get("lambda_l1", 100.0)),
            lambda_perc=float(loss_cfg.get("lambda_perc", 10.0)),
            lambda_ssim=float(loss_cfg.get("lambda_ssim", 3.0)),
        ).to(self.device)

        amp_enabled = bool(training_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=amp_enabled)

        self.logger = TrainingLogger(
            log_dir=str(self.log_dir),
            use_wandb=bool(config.get("logging", {}).get("use_wandb", False)),
            project_name=config.get("logging", {}).get(
                "project_name", "InfraNova-AI"
            ),
        )

        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=str(self.checkpoint_dir),
            monitor="val_ssim",
            mode="max",
        )

        self.early_stopping = EarlyStopping(
            patience=int(training_cfg.get("patience", 20)),
            monitor="val_ssim",
            mode="max",
        )

        self.scheduler_g = LinearLRScheduler(
            self.optimizer_g,
            total_epochs=self.total_epochs,
            decay_start_epoch=int(training_cfg.get("decay_start_epoch", 100)),
        )
        self.scheduler_d = LinearLRScheduler(
            self.optimizer_d,
            total_epochs=self.total_epochs,
            decay_start_epoch=int(training_cfg.get("decay_start_epoch", 100)),
        )

        self.start_epoch = 0
        self.history: Dict[str, list] = {
            "g_loss": [],
            "d_loss": [],
            "l1": [],
            "adv": [],
            "perc": [],
            "ssim": [],
            "val_psnr": [],
            "val_ssim": [],
        }

    @staticmethod
    def _to_device(
        batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    @staticmethod
    def _denorm(x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(-1, 1) + 1.0) / 2.0

    @staticmethod
    def _psnr(
        fake: torch.Tensor, real: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        mse = F.mse_loss(fake, real, reduction="none")
        mse = mse.mean(dim=(1, 2, 3)).clamp_min(eps)
        psnr = 10.0 * torch.log10(1.0 / mse)
        return psnr.mean()

    @staticmethod
    def _ssim_simple(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = fake.mean(dim=(2, 3), keepdim=True)
        mu_y = real.mean(dim=(2, 3), keepdim=True)

        sigma_x = ((fake - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_y = ((real - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_xy = ((fake - mu_x) * (real - mu_y)).mean(dim=(2, 3), keepdim=True)

        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
        )
        return ssim.mean()

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        running = {
            "g_loss": 0.0,
            "d_loss": 0.0,
            "l1": 0.0,
            "adv": 0.0,
            "perc": 0.0,
            "ssim": 0.0,
        }

        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise ValueError(
                "Training loader produced zero batches. Check the dataset split and batch size."
            )
        d_loss_last = torch.tensor(0.0, device=self.device)

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch, self.device)
            ir = batch["ir"]
            rgb = batch["rgb"]

            # ---- Train Discriminator ----
            self.model.discriminator.requires_grad_(True)
            self.optimizer_d.zero_grad(set_to_none=True)

            with autocast(self.device.type, enabled=self.scaler.is_enabled()):
                fake_rgb = self.model.generate(ir).detach()

                # Add noise to prevent discriminator overpowering
                noise_std = max(0.1 * (1 - epoch / self.total_epochs), 0.01)
                ir_noisy = ir + torch.randn_like(ir) * noise_std
                rgb_noisy = rgb + torch.randn_like(rgb) * noise_std
                fake_rgb_noisy = fake_rgb + torch.randn_like(fake_rgb) * noise_std

                real_pred = self.model.discriminate(ir_noisy, rgb_noisy)
                fake_pred = self.model.discriminate(ir_noisy, fake_rgb_noisy)

                real_loss = self.criterion.gan_loss(real_pred, True)
                fake_loss = self.criterion.gan_loss(fake_pred, False)
                d_loss = 0.5 * (real_loss + fake_loss)

            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.optimizer_d)
            torch.nn.utils.clip_grad_norm_(
                self.model.discriminator.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer_d)
            d_loss_last = d_loss.detach()

            # ---- Train Generator (NO noise) ----
            self.optimizer_g.zero_grad(set_to_none=True)
            self.model.discriminator.requires_grad_(False)

            try:
                with autocast(self.device.type, enabled=self.scaler.is_enabled()):
                    fake_rgb = self.model.generate(ir)
                    fake_pred_for_g = self.model.discriminate(ir, fake_rgb)

                    losses = self.criterion(
                        disc_fake_pred=fake_pred_for_g,
                        fake_rgb=fake_rgb,
                        real_rgb=rgb,
                    )
                    g_loss = losses["total"]

                self.scaler.scale(g_loss).backward()
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(
                    self.model.generator.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer_g)
            finally:
                self.model.discriminator.requires_grad_(True)

            self.scaler.update()

            running["g_loss"] += float(g_loss.detach().item())
            running["d_loss"] += float(d_loss_last.item())
            running["l1"] += float(losses["l1"].detach().item())
            running["adv"] += float(losses["adv"].detach().item())
            running["perc"] += float(losses["perc"].detach().item())
            running["ssim"] += float(losses["ssim"].detach().item())

        for key in running:
            running[key] /= max(num_batches, 1)

        return running

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        psnr_sum = 0.0
        ssim_sum = 0.0
        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            batch = self._to_device(batch, self.device)
            ir = batch["ir"]
            rgb = batch["rgb"]

            fake_rgb = self.model.generate(ir)

            fake_rgb_01 = self._denorm(fake_rgb)
            rgb_01 = self._denorm(rgb)

            psnr_sum += float(self._psnr(fake_rgb_01, rgb_01).item())
            ssim_sum += float(self._ssim_simple(fake_rgb_01, rgb_01).item())

        return {
            "val_psnr": psnr_sum / max(num_batches, 1),
            "val_ssim": ssim_sum / max(num_batches, 1),
        }

    @torch.no_grad()
    def _save_sample_images(self, epoch: int) -> None:
        """Save a grid of IR / Generated / Real RGB samples for visual inspection.

        This is purely diagnostic — any failure here is logged as a warning
        and never propagated so that training is not interrupted.
        """
        import matplotlib.pyplot as plt

        self.model.eval()
        fig = None

        try:
            if len(self.val_loader) == 0:
                return

            try:
                batch = next(iter(self.val_loader))
            except StopIteration:
                return

            batch = self._to_device(batch, self.device)
            sample_count = min(4, batch["ir"].size(0))
            if sample_count == 0:
                return

            ir = batch["ir"][:sample_count]
            rgb = batch["rgb"][:sample_count]

            fake_rgb = self.model.generate(ir)

            ir = self._denorm(ir).cpu()
            rgb = self._denorm(rgb).cpu()
            fake_rgb = self._denorm(fake_rgb).cpu()

            fig, axes = plt.subplots(
                sample_count, 3, figsize=(10, 3 * sample_count), squeeze=False,
            )

            for i in range(sample_count):
                axes[i, 0].imshow(ir[i][0], cmap="gray")
                axes[i, 0].set_title("IR")
                axes[i, 0].axis("off")

                generated = fake_rgb[i].numpy()
                generated = (
                    np.transpose(generated, (1, 2, 0)).clip(0, 1)
                    if generated.shape[0] == 3
                    else generated[0].clip(0, 1)
                )
                axes[i, 1].imshow(generated)
                axes[i, 1].set_title("Generated")
                axes[i, 1].axis("off")

                target = rgb[i].numpy()
                target = (
                    np.transpose(target, (1, 2, 0)).clip(0, 1)
                    if target.shape[0] == 3
                    else target[0].clip(0, 1)
                )
                axes[i, 2].imshow(target)
                axes[i, 2].set_title("Real RGB")
                axes[i, 2].axis("off")

            plt.tight_layout()
            save_path = self.visual_dir / f"epoch_{epoch}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        except Exception as exc:
            print(f"[WARNING] Could not save sample images for epoch {epoch}: {exc}")
        finally:
            if fig is not None:
                plt.close(fig)

    def fit(self, num_epochs: int) -> Dict[str, list]:
        """Train end-to-end with scheduler, checkpointing, and early stopping."""
        for epoch in range(self.start_epoch, num_epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()

            epoch_metrics = {**train_metrics, **val_metrics}

            self.logger.log_epoch(epoch + 1, epoch_metrics)

            for key, val in epoch_metrics.items():
                if key in self.history:
                    self.history[key].append(val)

            optimizers = {
                "generator": self.optimizer_g,
                "discriminator": self.optimizer_d,
            }

            self.checkpoint.step(
                model=self.model,
                optimizers=optimizers,
                epoch=epoch + 1,
                metrics=epoch_metrics,
                scaler=self.scaler,
            )

            if (epoch + 1) % self.sample_every == 0:
                self._save_sample_images(epoch + 1)

            self.scheduler_g.step(epoch + 1)
            self.scheduler_d.step(epoch + 1)

            current_lr = self.scheduler_g.get_last_lr()[0]

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"G: {train_metrics['g_loss']:.4f} | "
                f"D: {train_metrics['d_loss']:.4f} | "
                f"PSNR: {val_metrics['val_psnr']:.2f} | "
                f"SSIM: {val_metrics['val_ssim']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if self.early_stopping.step(epoch_metrics):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        self.logger.save_plot()
        return self.history
