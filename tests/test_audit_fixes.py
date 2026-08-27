"""
Regression & Robustness Test Suite for InfraNova AI Bug-Fix Audit.

Validates:
1. 2-Channel thermal input tensor conversion in numpy_to_tensor
2. Numerical robustness against constant scenes, NaNs, Infs, and uint16/float32 dtypes
3. Tiled sliding-window whole-raster inference across arbitrary & edge-case dimensions
4. API request validation and status codes (400 on empty/malformed, 200 on valid)
5. Model contract (shapes, parameter counts, finite output values)
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from demo.utils import _normalize_to_uint8, preprocess_ir_image
from src.inference.tiled_inference import TiledRasterInference
from src.models.pix2pix.discriminator import MultiScaleDiscriminator
from src.models.pix2pix.generator_hd import Pix2PixHDGenerator
from src.utils.image_processing import numpy_to_tensor, to_single_band_array


class TestImageProcessingRegression:
    """Test 2-channel tensor conversion and numerical safety."""

    def test_numpy_to_tensor_2channel_channel_first(self):
        arr = np.random.randn(2, 128, 128).astype(np.float32)
        tensor = numpy_to_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 128, 128)

    def test_numpy_to_tensor_2channel_channel_last(self):
        arr = np.random.randn(128, 128, 2).astype(np.float32)
        tensor = numpy_to_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 128, 128)

    def test_numpy_to_tensor_2channel_batch_bchw(self):
        arr = np.random.randn(4, 2, 128, 128).astype(np.float32)
        tensor = numpy_to_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4, 2, 128, 128)

    def test_numpy_to_tensor_2channel_batch_bhwc(self):
        arr = np.random.randn(4, 128, 128, 2).astype(np.float32)
        tensor = numpy_to_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4, 2, 128, 128)

    def test_to_single_band_array_with_nans(self):
        arr = np.array([[1.0, np.nan], [np.inf, 2.0]], dtype=np.float32)
        clean = to_single_band_array(arr)
        assert np.isfinite(clean).all()
        assert clean.shape == (2, 2)

    def test_normalize_to_uint8_constant_scene(self):
        arr = np.full((128, 128), 300.0, dtype=np.float32)
        norm = _normalize_to_uint8(arr)
        assert norm.shape == (128, 128)
        assert norm.dtype == np.uint8
        assert not np.isnan(norm).any()
        assert (norm == 0).all()

    def test_normalize_to_uint8_nan_scene(self):
        arr = np.full((128, 128), np.nan, dtype=np.float32)
        norm = _normalize_to_uint8(arr)
        assert norm.shape == (128, 128)
        assert norm.dtype == np.uint8
        assert not np.isnan(norm).any()
        assert (norm == 0).all()

    def test_preprocess_ir_image_2channel_tuple(self):
        b10 = np.random.uniform(250, 320, (128, 128)).astype(np.float32)
        b11 = np.random.uniform(250, 320, (128, 128)).astype(np.float32)
        tensor = preprocess_ir_image((b10, b11), image_size=128, target_channels=2)
        assert tensor.shape == (1, 2, 128, 128)
        assert torch.isfinite(tensor).all()
        assert tensor.min() >= -1.0 and tensor.max() <= 1.0


class TestModelContract:
    """Verify generator & discriminator architecture parameters and contracts."""

    def test_pix2pixhd_generator_contract(self):
        gen = Pix2PixHDGenerator(in_channels=2, out_channels=3)
        total_params = sum(p.numel() for p in gen.parameters())
        assert total_params == 21_383_238

        x = torch.randn(2, 2, 128, 128)
        out = gen(x)
        assert out.shape == (2, 3, 128, 128)
        assert torch.isfinite(out).all()

    def test_multiscale_discriminator_contract(self):
        disc = MultiScaleDiscriminator(in_channels=5, num_scales=2)
        total_params = sum(p.numel() for p in disc.parameters())
        assert total_params == 5_533_570

        x = torch.randn(2, 5, 128, 128)
        pred = disc(x)
        assert isinstance(pred, dict)
        assert "scale_0" in pred and "scale_1" in pred


class TestTiledInferenceEdgeCases:
    """Test whole-raster tiling on edge-case shapes."""

    @pytest.fixture(scope="class")
    @classmethod
    def tiled_engine(cls):
        return TiledRasterInference(tile_size=128, overlap=32, batch_size=4)

    def test_tiled_small_scene_padding(self, tiled_engine):
        b10 = np.random.uniform(0.2, 0.8, (64, 64)).astype(np.float32)
        b11 = np.random.uniform(0.2, 0.8, (64, 64)).astype(np.float32)
        res = tiled_engine.predict_scene(b10, b11)
        assert res["rgb_uint8"].shape == (64, 64, 3)
        assert not np.isnan(res["rgb_float"]).any()

    def test_tiled_odd_dimension_scene(self, tiled_engine):
        b10 = np.random.uniform(0.2, 0.8, (153, 147)).astype(np.float32)
        b11 = np.random.uniform(0.2, 0.8, (153, 147)).astype(np.float32)
        res = tiled_engine.predict_scene(b10, b11)
        assert res["rgb_uint8"].shape == (153, 147, 3)
        assert not np.isnan(res["rgb_float"]).any()

    def test_tiled_constant_scene(self, tiled_engine):
        b10 = np.full((200, 200), 300.0, dtype=np.float32)
        b11 = np.full((200, 200), 300.0, dtype=np.float32)
        res = tiled_engine.predict_scene(b10, b11)
        assert res["rgb_uint8"].shape == (200, 200, 3)
        assert not np.isnan(res["rgb_float"]).any()


class TestAPIEndpoints:
    """Test API endpoints for validation, status codes, and edge cases."""

    @pytest.fixture(scope="class")
    @classmethod
    def client(cls):
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/colorize",
            files={"file": ("empty.tif", b"", "image/tiff")},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_unsupported_file_type_returns_400(self, client):
        resp = client.post(
            "/colorize",
            files={"file": ("document.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()

    def test_valid_npy_colorize(self, client):
        arr = np.random.uniform(250, 320, (2, 128, 128)).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)

        resp = client.post(
            "/colorize",
            files={"file": ("dual_band.npy", buf.getvalue(), "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        img = Image.open(io.BytesIO(resp.content))
        assert img.size == (128, 128)
        assert img.mode == "RGB"

    def test_thermal_preview_endpoint(self, client):
        arr = np.random.uniform(250, 320, (128, 128)).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)

        resp = client.post(
            "/thermal-preview",
            files={"file": ("thermal.npy", buf.getvalue(), "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_clahe_postprocess_endpoint(self, client):
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/postprocess/clahe",
            files={"file": ("rgb.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
