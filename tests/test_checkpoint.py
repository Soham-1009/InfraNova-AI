"""Tests for checkpoint save/load."""

from __future__ import annotations

import pytest
import torch

from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_checkpoint, load_torch_checkpoint, save_checkpoint


class TestCheckpoint:
    """Tests for checkpoint save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve model weights."""
        model = Pix2Pix(in_channels=1, out_channels=3)
        optimizer = {
            "generator": torch.optim.Adam(model.generator.parameters(), lr=2e-4),
            "discriminator": torch.optim.Adam(model.discriminator.parameters(), lr=1e-4),
        }
        metrics = {"val_ssim": 0.85, "val_psnr": 28.5}
        path = str(tmp_path / "test.pth")

        save_checkpoint(model, optimizer, epoch=5, metrics=metrics, path=path)

        # Load into a new model
        model2 = Pix2Pix(in_channels=1, out_channels=3)
        optimizer2 = {
            "generator": torch.optim.Adam(model2.generator.parameters(), lr=2e-4),
            "discriminator": torch.optim.Adam(model2.discriminator.parameters(), lr=1e-4),
        }

        epoch, loaded_metrics = load_checkpoint(path, model2, optimizer2)

        assert epoch == 5
        assert loaded_metrics["val_ssim"] == pytest.approx(0.85)
        assert loaded_metrics["val_psnr"] == pytest.approx(28.5)

        # Verify weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_arch_info_saved(self, tmp_path):
        """Checkpoint should contain arch_info metadata."""
        model = Pix2Pix(in_channels=1, out_channels=3)
        optimizer = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "test.pth")

        save_checkpoint(model, optimizer, epoch=1, metrics={}, path=path)

        ckpt = load_torch_checkpoint(path)
        assert "arch_info" in ckpt
        info = ckpt["arch_info"]
        assert info["model"] == "Pix2Pix"
        assert "input_channels" in info
        assert "output_channels" in info
        assert "git_version" in info

    def test_load_legacy_checkpoint(self, tmp_path):
        """Legacy checkpoint without arch_info should still load."""
        model = Pix2Pix(in_channels=1, out_channels=3)
        optimizer = {
            "generator": torch.optim.Adam(model.generator.parameters()),
            "discriminator": torch.optim.Adam(model.discriminator.parameters()),
        }

        # Save without arch_info (legacy format)
        ckpt = {
            "epoch": 3,
            "metrics": {"val_ssim": 0.80},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {
                "generator": optimizer["generator"].state_dict(),
                "discriminator": optimizer["discriminator"].state_dict(),
            },
        }
        path = str(tmp_path / "legacy.pth")
        torch.save(ckpt, path)

        # Should load with a warning, not an error
        model2 = Pix2Pix(in_channels=1, out_channels=3)
        optimizer2 = {
            "generator": torch.optim.Adam(model2.generator.parameters()),
            "discriminator": torch.optim.Adam(model2.discriminator.parameters()),
        }
        epoch, metrics = load_checkpoint(path, model2, optimizer2)
        assert epoch == 3

    def test_mismatched_architecture_raises(self, tmp_path):
        """Loading checkpoint with wrong architecture should raise ValueError."""
        model = Pix2Pix(in_channels=1, out_channels=3)
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint with specific arch_info
        ckpt = {
            "epoch": 1,
            "metrics": {},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "arch_info": {
                "input_channels": 5,  # Intentionally wrong
                "output_channels": 3,
            },
        }
        path = str(tmp_path / "mismatch.pth")
        torch.save(ckpt, path)

        model2 = Pix2Pix(in_channels=1, out_channels=3)
        optimizer2 = torch.optim.Adam(model2.parameters())

        with pytest.raises(ValueError, match="mismatch"):
            load_checkpoint(path, model2, optimizer2)
