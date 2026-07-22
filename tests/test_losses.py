"""Tests for training losses."""

from __future__ import annotations

import pytest
import torch

from src.training.losses import CombinedLoss


class TestCombinedLoss:
    """Tests for the CombinedLoss module."""

    def test_forward_returns_dict(self, dummy_rgb_batch, device):
        """Loss forward should return a dict with expected keys."""
        criterion = CombinedLoss(device=device)

        fake_rgb = dummy_rgb_batch.clone().requires_grad_(True)
        real_rgb = dummy_rgb_batch.clone()
        # Create a fake discriminator prediction
        disc_pred = torch.randn(4, 1, 30, 30)

        losses = criterion(
            disc_fake_pred=disc_pred,
            fake_rgb=fake_rgb,
            real_rgb=real_rgb,
        )

        assert isinstance(losses, dict)
        assert "total" in losses
        assert "l1" in losses
        assert "adv" in losses
        assert "perc" in losses
        assert "ssim" in losses

    def test_loss_is_finite(self, dummy_rgb_batch, device):
        """All loss components should be finite."""
        criterion = CombinedLoss(device=device)

        fake_rgb = dummy_rgb_batch.clone().requires_grad_(True)
        real_rgb = dummy_rgb_batch.clone()
        disc_pred = torch.randn(4, 1, 30, 30)

        losses = criterion(
            disc_fake_pred=disc_pred,
            fake_rgb=fake_rgb,
            real_rgb=real_rgb,
        )

        for key, value in losses.items():
            assert torch.isfinite(value), f"Loss '{key}' is not finite: {value}"

    def test_loss_positive(self, dummy_rgb_batch, device):
        """Total loss should be positive."""
        criterion = CombinedLoss(device=device)

        fake_rgb = dummy_rgb_batch.clone().requires_grad_(True)
        real_rgb = dummy_rgb_batch.clone()
        disc_pred = torch.randn(4, 1, 30, 30)

        losses = criterion(
            disc_fake_pred=disc_pred,
            fake_rgb=fake_rgb,
            real_rgb=real_rgb,
        )

        assert losses["total"].item() > 0

    def test_gan_loss_real(self, device):
        """GAN loss for real labels should be computed."""
        criterion = CombinedLoss(device=device)
        pred = torch.randn(4, 1, 30, 30)
        loss = criterion.gan_loss(pred, True)
        assert torch.isfinite(loss)

    def test_gan_loss_fake(self, device):
        """GAN loss for fake labels should be computed."""
        criterion = CombinedLoss(device=device)
        pred = torch.randn(4, 1, 30, 30)
        loss = criterion.gan_loss(pred, False)
        assert torch.isfinite(loss)

    def test_identical_images_low_loss(self, dummy_rgb_batch, device):
        """Loss should be lower when predicted == target."""
        criterion = CombinedLoss(device=device)

        real_rgb = dummy_rgb_batch.clone()
        fake_rgb_same = real_rgb.clone().requires_grad_(True)
        fake_rgb_diff = torch.rand_like(real_rgb).requires_grad_(True)
        disc_pred = torch.randn(4, 1, 30, 30)

        loss_same = criterion(disc_fake_pred=disc_pred, fake_rgb=fake_rgb_same, real_rgb=real_rgb)
        loss_diff = criterion(disc_fake_pred=disc_pred, fake_rgb=fake_rgb_diff, real_rgb=real_rgb)

        # L1 should be lower for identical images
        assert loss_same["l1"].item() < loss_diff["l1"].item()
