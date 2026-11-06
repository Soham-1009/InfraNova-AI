"""
Security Audit Verification Suite for InfraNova AI.

Tests:
1. HTTP security headers (nosniff, DENY, CSP, HSTS, etc.)
2. File upload size limit enforcement (HTTP 413 Payload Too Large)
3. Excessive spatial dimension DoS protection (HTTP 400)
4. NumPy pickle deserialization injection protection across API and inference engine
5. CLAHE boundary parameter and file-type validation
6. Dual-band upload extension and shape consistency validation
7. Earth Engine download SSRF scheme and private-IP validation
8. Earth Engine zip bomb and zip slip prevention
9. Earth Engine destination filename traversal prevention
10. TIFF header decompression bomb early rejection
11. PyTorch checkpoint weights-only safe deserialization
"""

from __future__ import annotations

import io
import pickle
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main
from api.main import app
from scripts.download.download_landsat9 import download_and_extract
from src.inference.tiled_inference import TiledRasterInference
from src.utils.checkpoint import load_torch_checkpoint


class TestSecurityHardening:
    """Security verification tests for API, deserialization, and pipeline handlers."""

    @pytest.fixture
    def client(self):
        with TestClient(app) as c:
            yield c

    def test_http_security_headers_present(self, client):
        """Verify standard defense-in-depth security headers are attached."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
        assert "default-src 'self'" in resp.headers.get("content-security-policy", "")
        assert "max-age=31536000" in resp.headers.get("strict-transport-security", "")
        assert resp.headers.get("x-xss-protection") == "0"
        assert resp.headers.get("x-permitted-cross-domain-policies") == "none"
        assert "geolocation=()" in resp.headers.get("permissions-policy", "")

    def test_upload_file_exceeding_max_size_returns_413(self, client, monkeypatch):
        """Verify uploads exceeding max permitted bytes return HTTP 413 without loading into RAM."""
        monkeypatch.setattr(api_main, "MAX_UPLOAD_SIZE", 1024)  # 1 KB limit for testing

        oversized_data = b"X" * 2048
        resp = client.post(
            "/colorize",
            files={"file": ("large.png", oversized_data, "image/png")},
        )
        assert resp.status_code == 413
        assert "exceeds maximum permitted size" in resp.json()["detail"].lower()

    def test_excessive_spatial_dimensions_rejected(self, client, monkeypatch):
        """Verify images with excessive spatial dimensions (>MAX_IMAGE_DIM) are rejected."""
        monkeypatch.setattr(api_main, "MAX_IMAGE_DIM", 256)

        arr = np.random.uniform(250, 320, (512, 512)).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)

        resp = client.post(
            "/colorize",
            files={"file": ("oversized_dim.npy", buf.getvalue(), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "exceed maximum permitted dimension" in resp.json()["detail"].lower()

    def test_pickle_in_npy_is_rejected_in_colorize(self, client):
        """Verify uploaded .npy containing pickled Python objects is rejected with allow_pickle=False."""
        class MaliciousPayload:
            def __reduce__(self):
                return (eval, ("1+1",))

        obj_arr = np.array([MaliciousPayload()], dtype=object)
        buf = io.BytesIO()
        np.save(buf, obj_arr, allow_pickle=True)
        buf.seek(0)

        resp = client.post(
            "/colorize",
            files={"file": ("exploit.npy", buf.getvalue(), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "invalid input image" in resp.json()["detail"].lower()

    def test_pickle_in_npy_is_rejected_in_thermal_preview(self, client):
        """Verify .npy containing pickled Python objects is rejected in /thermal-preview."""
        class MaliciousPayload:
            def __reduce__(self):
                return (eval, ("1+1",))

        obj_arr = np.array([MaliciousPayload()], dtype=object)
        buf = io.BytesIO()
        np.save(buf, obj_arr, allow_pickle=True)
        buf.seek(0)

        resp = client.post(
            "/thermal-preview",
            files={"file": ("exploit.npy", buf.getvalue(), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "invalid input image" in resp.json()["detail"].lower()

    def test_tiled_raster_load_raster_rejects_pickle(self, tmp_path):
        """Verify TiledRasterInference.load_raster rejects pickled objects via allow_pickle=False."""
        class MaliciousPayload:
            def __reduce__(self):
                return (eval, ("1+1",))

        obj_arr = np.array([MaliciousPayload()], dtype=object)
        exploit_file = tmp_path / "exploit.npy"
        np.save(exploit_file, obj_arr, allow_pickle=True)

        with pytest.raises((ValueError, OSError)):
            TiledRasterInference.load_raster(exploit_file)

    def test_dual_band_extension_validation(self, client):
        """Verify band10 and band11 reject unapproved file extensions."""
        arr = np.random.uniform(250, 320, (128, 128)).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        raw = buf.getvalue()

        resp = client.post(
            "/colorize",
            files={
                "band10": ("b10.exe", raw, "application/octet-stream"),
                "band11": ("b11.npy", raw, "application/octet-stream"),
            },
        )
        assert resp.status_code == 400
        assert "unsupported file type" in resp.json()["detail"].lower()

    def test_dual_band_mismatched_dimensions_returns_400(self, client):
        """Verify mismatched band10 and band11 spatial dimensions return HTTP 400 rather than 500."""
        b10 = np.random.uniform(250, 320, (128, 128)).astype(np.float32)
        b11 = np.random.uniform(250, 320, (256, 256)).astype(np.float32)

        buf10 = io.BytesIO()
        np.save(buf10, b10)
        buf11 = io.BytesIO()
        np.save(buf11, b11)

        resp = client.post(
            "/colorize",
            files={
                "band10": ("b10.npy", buf10.getvalue(), "application/octet-stream"),
                "band11": ("b11.npy", buf11.getvalue(), "application/octet-stream"),
            },
        )
        assert resp.status_code == 400
        assert "does not match" in resp.json()["detail"].lower()

    def test_clahe_parameter_boundary_validation(self, client):
        """Verify CLAHE clip_limit and grid_size reject out-of-bounds parameters."""
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.getvalue()

        # clip_limit > 100
        resp = client.post(
            "/postprocess/clahe?clip_limit=150.0",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 400
        assert "clip_limit" in resp.json()["detail"]

        # negative clip_limit
        resp = client.post(
            "/postprocess/clahe?clip_limit=-1.0",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 400

        # grid_size > 128
        resp = client.post(
            "/postprocess/clahe?grid_size=256",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 400
        assert "grid_size" in resp.json()["detail"]

        # grid_size <= 0
        resp = client.post(
            "/postprocess/clahe?grid_size=0",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 400

    def test_clahe_unsupported_file_type_rejected(self, client):
        """Verify CLAHE rejects file types not suitable for RGB image processing (e.g. .npy, .pdf)."""
        arr = np.zeros((64, 64), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)

        resp = client.post(
            "/postprocess/clahe",
            files={"file": ("data.npy", buf.getvalue(), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "unsupported file type" in resp.json()["detail"].lower()

    def test_tiff_decompression_bomb_pre_check(self, client, monkeypatch):
        """Verify TIFFs with header dimensions exceeding MAX_IMAGE_DIM are rejected before full decode."""
        monkeypatch.setattr(api_main, "MAX_IMAGE_DIM", 256)

        arr = np.zeros((300, 300), dtype=np.uint8)
        buf = io.BytesIO()
        tifffile.imwrite(buf, arr)
        buf.seek(0)

        resp = client.post(
            "/colorize",
            files={"file": ("huge.tif", buf.getvalue(), "image/tiff")},
        )
        assert resp.status_code == 400
        assert "exceed maximum permitted dimension" in resp.json()["detail"].lower()

    def test_download_and_extract_rejects_insecure_url_schemes(self, tmp_path):
        """Verify non-HTTP/HTTPS URLs are rejected to prevent file:// SSRF."""
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            download_and_extract("file:///etc/passwd", tmp_path, "out.tif")

        with pytest.raises(ValueError, match="Invalid URL scheme"):
            download_and_extract("ftp://example.com/data.tif", tmp_path, "out.tif")

    def test_download_and_extract_rejects_ssrf_private_ips(self, tmp_path):
        """Verify access to loopback and private cloud metadata IP ranges is blocked."""
        with pytest.raises(ValueError, match="loopback"):
            download_and_extract("http://127.0.0.1:8000/data.tif", tmp_path, "out.tif")

        with pytest.raises(ValueError, match="loopback"):
            download_and_extract("http://localhost:8000/data.tif", tmp_path, "out.tif")

        with pytest.raises(ValueError, match="private/local network"):
            download_and_extract("http://169.254.169.254/latest/meta-data/", tmp_path, "out.tif")

        with pytest.raises(ValueError, match="private/local network"):
            download_and_extract("http://10.0.0.1/scene.tif", tmp_path, "out.tif")

    def test_download_and_extract_rejects_path_traversal_out_name(self, tmp_path):
        """Verify directory traversal in out_name argument is rejected."""
        with pytest.raises(ValueError, match=r"Invalid destination|traversal"):
            download_and_extract("https://earthengine.googleapis.com/download", tmp_path, "..")

    def test_download_and_extract_rejects_zip_slip(self, tmp_path):
        """Verify zip files with directory traversal paths are rejected."""
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            z.writestr("../../../evil.tif", b"FAKE_TIFF_DATA")
        zip_buf.seek(0)

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [zip_buf.getvalue()]
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp), pytest.raises(ValueError, match="Path traversal"):
            download_and_extract("https://earthengine.googleapis.com/download", tmp_path, "safe.tif")

    def test_download_and_extract_rejects_zip_bomb(self, tmp_path):
        """Verify zip files exceeding uncompressed safety limit trigger ValueError."""
        mock_info = MagicMock()
        mock_info.file_size = 600 * 1024 * 1024  # 600 MB
        mock_info.filename = "huge.tif"
        mock_info.is_dir.return_value = False

        mock_zip = MagicMock()
        mock_zip.infolist.return_value = [mock_info]

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"PK\x03\x04" + b"\x00" * 20]
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp), patch("zipfile.ZipFile") as mock_zipfile:
            mock_zipfile.return_value.__enter__.return_value = mock_zip
            with pytest.raises(ValueError, match="Zip archive uncompressed size"):
                download_and_extract("https://earthengine.googleapis.com/download", tmp_path, "safe.tif")

    def test_load_torch_checkpoint_weights_only_prevents_code_exec(self, tmp_path):
        """Verify load_torch_checkpoint with weights_only=True rejects arbitrary executable pickles."""
        class Exploit:
            def __reduce__(self):
                return (eval, ("2+2",))

        exploit_path = tmp_path / "exploit.pth"
        with open(exploit_path, "wb") as f:
            pickle.dump(Exploit(), f)

        with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError)):
            load_torch_checkpoint(exploit_path)
