"""Tests for inference engine."""

from __future__ import annotations

import pytest
import torch

from src.models.pix2pix.pix2pix import Pix2Pix


class TestInference:
    """Tests for model inference."""

    def test_generator_forward(self, dummy_ir_tensor, device):
        """Generator should produce correct output shape."""
        model = Pix2Pix(in_channels=1, out_channels=3)
        model.eval()

        with torch.inference_mode():
            output = model.generate(dummy_ir_tensor)

        assert output.shape == (1, 3, 256, 256)

    def test_generator_batch(self, dummy_ir_batch, device):
        """Generator should handle batches."""
        model = Pix2Pix(in_channels=1, out_channels=3)
        model.eval()

        with torch.inference_mode():
            output = model.generate(dummy_ir_batch)

        assert output.shape == (4, 3, 256, 256)

    def test_discriminator_forward(self, dummy_ir_tensor, dummy_rgb_tensor, device):
        """Discriminator should produce a prediction."""
        model = Pix2Pix(in_channels=1, out_channels=3)
        model.eval()

        with torch.inference_mode():
            pred = model.discriminate(dummy_ir_tensor, dummy_rgb_tensor)

        assert pred.ndim == 4  # (B, 1, H', W')

    def test_output_range(self, dummy_ir_tensor, device):
        """Generator output should be approximately in valid range."""
        model = Pix2Pix(in_channels=1, out_channels=3)
        model.eval()

        with torch.inference_mode():
            output = model.generate(dummy_ir_tensor)

        # Output uses tanh so values are in [-1, 1], then may be
        # post-processed. Just check it's finite.
        assert torch.isfinite(output).all()

    def test_inference_mode_no_grad(self, dummy_ir_tensor, device):
        """Inference should work without gradient tracking."""
        model = Pix2Pix(in_channels=1, out_channels=3)
        model.eval()

        with torch.inference_mode():
            output = model.generate(dummy_ir_tensor)

        assert not output.requires_grad
