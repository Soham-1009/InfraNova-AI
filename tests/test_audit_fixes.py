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
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main
from api.main import app
from demo.utils import _normalize_to_uint8, preprocess_ir_image
from src.inference.landsat_inference import LandsatColorizationInference
from src.inference.tiled_inference import TiledRasterInference
from src.models.pix2pix.discriminator import MultiScaleDiscriminator
from src.models.pix2pix.generator_hd import Pix2PixHDGenerator
from src.training.losses import SSIMLoss
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

    @pytest.fixture
    def client(self):
        with TestClient(app) as c:
            yield c

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


class TestAuditPonytailFixes:
    """Verify bug-audit fixes across security, API, pipelines, and loss calculations."""

    def test_dockerignore_and_gitignore_contain_keys(self):
        root = Path(__file__).resolve().parents[1]
        dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
        assert "*.key" in dockerignore
        assert "*.pem" in dockerignore

        gitignore = (root / ".gitignore").read_text(encoding="utf-8")
        assert "*.key" in gitignore
        assert "*.pem" in gitignore

    def test_cors_disables_credentials_for_wildcard(self):
        cors_mw = None
        for mw in api_main.app.user_middleware:
            if "CORSMiddleware" in str(mw):
                cors_mw = mw
                break
        assert cors_mw is not None
        if api_main.CORS_ORIGINS == ["*"]:
            assert cors_mw.kwargs.get("allow_credentials") is False

    def test_healthcheck_returns_degraded_on_missing_checkpoint(self, monkeypatch):
        monkeypatch.setattr(
            api_main,
            "get_engine",
            lambda: (_ for _ in ()).throw(FileNotFoundError("Checkpoint not found")),
        )
        with TestClient(api_main.app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["model_loaded"] is False

    def test_geotiff_multiband_upload(self):
        with TestClient(api_main.app) as client:
            b10 = np.full((128, 128), 290.0, dtype=np.float32)
            b11 = np.full((128, 128), 310.0, dtype=np.float32)
            dual_arr = np.stack([b10, b11], axis=0)

            buf = io.BytesIO()
            tifffile.imwrite(buf, dual_arr)
            buf.seek(0)

            resp = client.post(
                "/colorize",
                files={"file": ("SCENE.TIF", buf.getvalue(), "image/tiff")},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"
            out_img = Image.open(io.BytesIO(resp.content))
            assert out_img.size == (128, 128)
            assert out_img.mode == "RGB"

    def test_dual_upload_case_insensitive_npy(self):
        with TestClient(api_main.app) as client:
            b10 = np.random.uniform(250, 320, (128, 128)).astype(np.float32)
            b11 = np.random.uniform(250, 320, (128, 128)).astype(np.float32)

            buf10 = io.BytesIO()
            np.save(buf10, b10)
            buf11 = io.BytesIO()
            np.save(buf11, b11)

            resp = client.post(
                "/colorize",
                files={
                    "band10": ("BAND10.NPY", buf10.getvalue(), "application/octet-stream"),
                    "band11": ("BAND11.NPY", buf11.getvalue(), "application/octet-stream"),
                },
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"

    def test_landsat_inference_save_tiff_rgb_order(self, tmp_path):
        h, w = 16, 16
        r = np.full((h, w), 200, dtype=np.uint8)
        g = np.full((h, w), 100, dtype=np.uint8)
        b = np.full((h, w), 50, dtype=np.uint8)
        rgb_arr = np.stack([r, g, b], axis=-1)

        out_file = tmp_path / "rgb_test.tif"
        LandsatColorizationInference.save_tiff(rgb_arr, str(out_file))

        read_back = tifffile.imread(str(out_file))
        if read_back.shape[0] == 3:
            assert (read_back[0] == 200).all(), "Band 1 should be Red"
            assert (read_back[1] == 100).all(), "Band 2 should be Green"
            assert (read_back[2] == 50).all(), "Band 3 should be Blue"
        else:
            assert (read_back[:, :, 0] == 200).all(), "Band 1 should be Red"
            assert (read_back[:, :, 1] == 100).all(), "Band 2 should be Green"
            assert (read_back[:, :, 2] == 50).all(), "Band 3 should be Blue"

    def test_tiled_inference_sub_65px_scene(self):
        tiler = TiledRasterInference(tile_size=128, overlap=32, batch_size=2)
        b10 = np.random.uniform(0.2, 0.8, (48, 48)).astype(np.float32)
        b11 = np.random.uniform(0.2, 0.8, (48, 48)).astype(np.float32)
        res = tiler.predict_scene(b10, b11)
        assert res["rgb_uint8"].shape == (48, 48, 3)
        assert np.isfinite(res["rgb_float"]).all()

    def test_pipeline_runner_stages(self):
        from scripts.pipeline import run_pipeline

        source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
        assert "process_landsat_patches.py" in source
        assert "split_patches.py" in source
        assert "configs/exp9_resnet_generator.yaml" in source

    def test_ssim_loss_channel_update(self):
        loss_fn = SSIMLoss(window_size=11)
        assert loss_fn.channel == 3

        x = torch.randn(2, 1, 32, 32)
        y = torch.randn(2, 1, 32, 32)
        _ = loss_fn(x, y)
        assert loss_fn.channel == 1

    def test_export_model_dynamic_spatial_axes(self):
        source = Path("scripts/deployment/export/export_model.py").read_text(encoding="utf-8")
        assert '"height"' in source
        assert '"width"' in source

    def test_demo_inference_consolidated_onto_landsat_inference(self):
        import demo.inference as demo_inf
        import src.inference.landsat_inference as src_inf

        assert demo_inf.InferenceEngine is src_inf.InferenceEngine
        assert demo_inf.LandsatColorizationInference is src_inf.LandsatColorizationInference

        alias_text = Path(demo_inf.__file__).read_text(encoding="utf-8")
        assert len(alias_text.strip().splitlines()) <= 25

    def test_dockerfile_healthcheck_dynamic_port(self):
        content = Path("Dockerfile").read_text(encoding="utf-8")
        assert "http://localhost:${PORT:-8000}/health" in content

    def test_evaluate_ssim_simple_windowed(self):
        from scripts.evaluation.evaluate import compute_ssim_simple

        pred = np.full((32, 32, 3), 0.5, dtype=np.float32)
        target = np.full((32, 32, 3), 0.5, dtype=np.float32)
        score = compute_ssim_simple(pred, target)
        assert abs(score - 1.0) < 1e-4

        score_2d = compute_ssim_simple(pred[:, :, 0], target[:, :, 0])
        assert abs(score_2d - 1.0) < 1e-4

    def test_multichannel_geotiff_upload_over_2_channels(self):
        with TestClient(api_main.app) as client:
            b10 = np.full((128, 128), 290.0, dtype=np.float32)
            b11 = np.full((128, 128), 310.0, dtype=np.float32)
            b_extra = np.full((128, 128), 100.0, dtype=np.float32)
            multi_arr = np.stack([b10, b11, b_extra], axis=0)

            buf = io.BytesIO()
            tifffile.imwrite(buf, multi_arr)
            buf.seek(0)

            resp = client.post(
                "/colorize",
                files={"file": ("SCENE_3BAND.TIF", buf.getvalue(), "image/tiff")},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"
