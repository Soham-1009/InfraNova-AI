"""Shared test fixtures for InfraNova AI unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def device():
    """Return available device (prefer CPU for tests)."""
    return "cpu"


@pytest.fixture
def dummy_ir_tensor():
    """Single-channel thermal input tensor (1, 1, 256, 256)."""
    return torch.randn(1, 1, 256, 256)


@pytest.fixture
def dummy_rgb_tensor():
    """3-channel RGB tensor (1, 3, 256, 256) in [0, 1]."""
    return torch.rand(1, 3, 256, 256)


@pytest.fixture
def dummy_ir_batch():
    """Batch of thermal inputs (4, 1, 256, 256)."""
    return torch.randn(4, 1, 256, 256)


@pytest.fixture
def dummy_rgb_batch():
    """Batch of RGB targets (4, 3, 256, 256) in [0, 1]."""
    return torch.rand(4, 3, 256, 256)


@pytest.fixture
def tmp_checkpoint(tmp_path):
    """Create a temporary checkpoint file for testing."""
    from src.models.pix2pix.pix2pix import Pix2Pix

    model = Pix2Pix(in_channels=1, out_channels=3)
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=2e-4)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4)

    checkpoint = {
        "epoch": 10,
        "metrics": {"val_ssim": 0.85, "val_psnr": 28.5},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {
            "generator": optimizer_g.state_dict(),
            "discriminator": optimizer_d.state_dict(),
        },
        "arch_info": {
            "model": "Pix2Pix",
            "generator": "UNetGenerator",
            "discriminator": "PatchGANDiscriminator",
            "input_channels": 1,
            "output_channels": 3,
            "image_size": 256,
        },
    }

    ckpt_path = tmp_path / "test_checkpoint.pth"
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    """Create a temporary dataset directory with fake samples."""
    split_dir = tmp_path / "splits" / "test"
    split_dir.mkdir(parents=True)

    for i in range(3):
        sample_dir = split_dir / f"region_sample_{i:03d}_00000"
        sample_dir.mkdir()

        np.save(sample_dir / "tir_200m.npy", np.random.rand(64, 64).astype(np.float32))
        np.save(sample_dir / "tir_100m.npy", np.random.rand(128, 128).astype(np.float32))
        np.save(sample_dir / "rgb_100m.npy", np.random.rand(3, 128, 128).astype(np.float32))

    return tmp_path / "splits"
