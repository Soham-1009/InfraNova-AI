"""Tests for Landsat9Dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.datasets.landsat9_dataset import Landsat9Dataset


class TestLandsat9Dataset:
    """Tests for dataset loading, shapes, and augmentation."""

    def test_dataset_loads(self, tmp_dataset_dir):
        """Dataset should load without errors."""
        dataset = Landsat9Dataset(
            root_dir=str(tmp_dataset_dir),
            split="test",
            image_size=128,
            augment=False,
        )
        assert len(dataset) == 3

    def test_sample_keys(self, tmp_dataset_dir):
        """Each sample should have 'ir' and 'rgb' keys."""
        dataset = Landsat9Dataset(
            root_dir=str(tmp_dataset_dir),
            split="test",
            image_size=128,
            augment=False,
        )
        sample = dataset[0]
        assert "ir" in sample or "input" in sample or "tir" in sample
        assert "rgb" in sample or "target" in sample

    def test_tensor_shapes(self, tmp_dataset_dir):
        """Tensors should have correct shapes."""
        dataset = Landsat9Dataset(
            root_dir=str(tmp_dataset_dir),
            split="test",
            image_size=128,
            augment=False,
        )
        sample = dataset[0]

        # Get IR tensor (could be under different keys)
        ir = sample.get("ir", sample.get("tir", sample.get("input")))
        rgb = sample.get("rgb", sample.get("target"))

        assert isinstance(ir, torch.Tensor)
        assert isinstance(rgb, torch.Tensor)
        assert ir.ndim == 3  # (C, H, W)
        assert rgb.ndim == 3

    def test_missing_split_raises(self, tmp_dataset_dir):
        """Non-existent split should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Landsat9Dataset(
                root_dir=str(tmp_dataset_dir),
                split="nonexistent",
                image_size=128,
            )

    def test_no_augmentation_on_test(self, tmp_dataset_dir):
        """Augmentation should be disabled for test split."""
        dataset = Landsat9Dataset(
            root_dir=str(tmp_dataset_dir),
            split="test",
            image_size=128,
            augment=True,  # Request augmentation
        )
        # Should be False because split != "train"
        assert not dataset.augment
