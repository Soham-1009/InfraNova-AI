"""Tests for preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.image_processing import (
    tensor_to_numpy,
    numpy_to_tensor,
)


class TestPreprocessing:
    """Tests for image preprocessing and conversion utilities."""

    def test_tensor_to_numpy_shape(self):
        """tensor_to_numpy should convert (C, H, W) to (H, W, C)."""
        tensor = torch.rand(3, 64, 64)
        arr = tensor_to_numpy(tensor)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (64, 64, 3)

    def test_numpy_to_tensor_shape(self):
        """numpy_to_tensor should convert (H, W, C) to (C, H, W)."""
        arr = np.random.rand(64, 64, 3).astype(np.float32)
        tensor = numpy_to_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_roundtrip(self):
        """tensor → numpy → tensor should preserve values."""
        original = torch.rand(3, 32, 32)
        arr = tensor_to_numpy(original)
        recovered = numpy_to_tensor(arr)
        assert torch.allclose(original, recovered, atol=1e-6)

    def test_single_channel(self):
        """Should handle single-channel images."""
        tensor = torch.rand(1, 64, 64)
        arr = tensor_to_numpy(tensor)
        assert arr.shape == (64, 64, 1)

    def test_batch_dimension(self):
        """Should handle 4D tensors with batch dim."""
        tensor = torch.rand(2, 3, 64, 64)
        # tensor_to_numpy should handle batch
        arr = tensor_to_numpy(tensor)
        # Implementation may vary; just check it doesn't crash
        assert isinstance(arr, np.ndarray)


class TestSeedEverything:
    """Tests for reproducibility utilities."""

    def test_seed_determinism(self):
        """Same seed should produce same random values."""
        from src.utils.seed import seed_everything

        seed_everything(42)
        a = torch.randn(10)

        seed_everything(42)
        b = torch.randn(10)

        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds should produce different values."""
        from src.utils.seed import seed_everything

        seed_everything(42)
        a = torch.randn(10)

        seed_everything(99)
        b = torch.randn(10)

        assert not torch.equal(a, b)

    def test_numpy_determinism(self):
        """NumPy should also be deterministic."""
        from src.utils.seed import seed_everything

        seed_everything(42)
        a = np.random.rand(10)

        seed_everything(42)
        b = np.random.rand(10)

        np.testing.assert_array_equal(a, b)
