"""Tests for Pix2PixHD inference engine."""

from __future__ import annotations

import torch

from src.models.pix2pix.pix2pix import Pix2Pix


class TestInference:
    """Tests for model inference."""

    def test_generator_forward(self, dummy_ir_tensor, device):
        """Generator should produce correct output shape."""
        model = Pix2Pix(in_channels=2, out_channels=3, image_size=128, num_scales=2)
        model.eval()

        with torch.inference_mode():
            output = model.generate(dummy_ir_tensor)

        assert output.shape == (1, 3, 128, 128)
        assert torch.isfinite(output).all()

    def test_generator_batch(self, dummy_ir_batch, device):
        """Generator should handle batches."""
        model = Pix2Pix(in_channels=2, out_channels=3, image_size=128, num_scales=2)
        model.eval()

        with torch.inference_mode():
            output = model.generate(dummy_ir_batch)

        assert output.shape == (4, 3, 128, 128)
        assert torch.isfinite(output).all()

    def test_discriminator_forward(self, dummy_ir_tensor, dummy_rgb_tensor, device):
        """Discriminator should produce multi-scale predictions."""
        model = Pix2Pix(in_channels=2, out_channels=3, image_size=128, num_scales=2)
        model.eval()

        with torch.inference_mode():
            pred = model.discriminate(dummy_ir_tensor, dummy_rgb_tensor)

        assert isinstance(pred, dict)
        assert "scale_0" in pred
        assert "scale_1" in pred
        assert pred["scale_0"].ndim == 4

    def test_output_range(self, dummy_ir_tensor, device):
        """Generator output should be bounded and finite."""
        model = Pix2Pix(in_channels=2, out_channels=3, image_size=128, num_scales=2)
        model.eval()

        with torch.inference_mode():
            output = model.generate(dummy_ir_tensor)

        assert torch.isfinite(output).all()
        assert (output >= -1.0).all() and (output <= 1.0).all()

    def test_inference_mode_no_grad(self, dummy_ir_tensor, device):
        """Inference should work without gradient tracking."""
        model = Pix2Pix(in_channels=2, out_channels=3, image_size=128, num_scales=2)
        model.eval()

        with torch.inference_mode():
            output = model.generate(dummy_ir_tensor)

        assert not output.requires_grad
