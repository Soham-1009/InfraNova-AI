"""Tests for Pix2PixHD inference engine."""

from __future__ import annotations

import numpy as np
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

    def test_inference_engine_with_torch_device_and_tuple_input(self, tmp_checkpoint):
        """InferenceEngine accepts torch.device and tuple of numpy arrays."""
        from PIL import Image

        from demo.inference import InferenceEngine

        engine = InferenceEngine(
            checkpoint_path=str(tmp_checkpoint),
            device=torch.device("cpu"),
            image_size=128,
        )
        assert engine.device == torch.device("cpu")
        b10 = np.random.uniform(250, 320, (128, 128)).astype(np.float32)
        b11 = np.random.uniform(250, 320, (128, 128)).astype(np.float32)
        out = engine.predict((b10, b11), use_tta=False)
        assert isinstance(out, Image.Image)
        assert out.size == (128, 128)

        # Batch predict with tuple
        batch_out = engine.predict_batch([(b10, b11)], use_tta=False)
        assert len(batch_out) == 1
        assert isinstance(batch_out[0], Image.Image)

    def test_tiled_inference_with_torch_device(self, tmp_checkpoint):
        """TiledRasterInference accepts torch.device and runs predict_scene."""
        from src.inference.tiled_inference import TiledRasterInference

        tiler = TiledRasterInference(
            checkpoint_path=str(tmp_checkpoint),
            tile_size=128,
            overlap=32,
            batch_size=2,
            device=torch.device("cpu"),
        )
        assert tiler.tile_size == 128
        assert tiler.overlap == 32
        assert tiler.batch_size == 2
        b10 = np.random.uniform(250, 320, (160, 160)).astype(np.float32)
        b11 = np.random.uniform(250, 320, (160, 160)).astype(np.float32)
        res = tiler.predict_scene(b10, b11, use_tta=True)
        assert res["rgb_uint8"].shape == (160, 160, 3)
