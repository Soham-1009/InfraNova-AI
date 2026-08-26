import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app

def test_full_api_flow():
    print("=" * 80)
    print("TESTING FULL E2E WEB & API INTEGRATION (REACT -> FASTAPI -> PIX2PIXHD)")
    print("=" * 80)

    with TestClient(app) as client:
        # 1. Health check
        res_health = client.get("/health")
        print("1. /health Endpoint:")
        print(f"   Status: {res_health.status_code}")
        print(f"   JSON: {res_health.json()}")
        assert res_health.status_code == 200
        assert res_health.json()["model_loaded"] is True

        # 2. React UI Serving check
        res_ui = client.get("/")
        print("\n2. Root / Endpoint (React UI Serving):")
        print(f"   Status: {res_ui.status_code}")
        print(f"   Content-Type: {res_ui.headers.get('content-type')}")
        assert res_ui.status_code == 200
        assert "<div id=\"root\">" in res_ui.text

        # 3. Real dual-band Landsat 9 TIFF Upload
        scene_dir = PROJECT_ROOT / "data" / "landsat9" / "raw" / "accra"
        b11_scene_dir = PROJECT_ROOT / "data" / "landsat9_b10_b11" / "raw" / "accra"
        b10_bytes = (scene_dir / "tir.tif").read_bytes()
        b11_bytes = (b11_scene_dir / "tir_b11.tif").read_bytes()

        print("\n3. /colorize with Real Landsat 9 B10 + B11 TIFF files:")
        print(f"   B10 size: {len(b10_bytes):,} bytes | B11 size: {len(b11_bytes):,} bytes")
        
        res_tiff = client.post(
            "/colorize",
            files={
                "band10": ("tir.tif", io.BytesIO(b10_bytes), "image/tiff"),
                "band11": ("tir_b11.tif", io.BytesIO(b11_bytes), "image/tiff"),
            },
        )
        print(f"   Status: {res_tiff.status_code}")
        print(f"   Content-Type: {res_tiff.headers.get('content-type')}")
        print(f"   Payload size: {len(res_tiff.content):,} bytes")
        assert res_tiff.status_code == 200
        assert res_tiff.headers.get("content-type") == "image/png"
        
        # Verify valid PIL Image
        img_out = Image.open(io.BytesIO(res_tiff.content))
        print(f"   Synthesized RGB size: {img_out.size}, mode: {img_out.mode}")
        assert img_out.size == (128, 128)
        assert img_out.mode == "RGB"

        # 4. /colorize with TTA enabled
        print("\n4. /colorize with Test-Time Augmentation (TTA):")
        res_tta = client.post(
            "/colorize?tta=true",
            files={
                "band10": ("tir.tif", io.BytesIO(b10_bytes), "image/tiff"),
                "band11": ("tir_b11.tif", io.BytesIO(b11_bytes), "image/tiff"),
            },
        )
        print(f"   Status: {res_tta.status_code}")
        print(f"   Payload size: {len(res_tta.content):,} bytes")
        assert res_tta.status_code == 200

        # 5. Real 2-channel .npy Test Patch Upload
        patch_dir = PROJECT_ROOT / "data" / "landsat9_b10_b11" / "splits" / "test" / "adilabad_sample_000_12262"
        b10_arr = np.load(patch_dir / "tir_100m.npy")
        b11_arr = np.load(patch_dir / "tir_b11_100m.npy")
        npy_2ch = np.stack([b10_arr, b11_arr], axis=0)
        buf_npy = io.BytesIO()
        np.save(buf_npy, npy_2ch)
        buf_npy.seek(0)

        print("\n5. /predict with 2-channel .npy test patch:")
        res_npy = client.post(
            "/predict",
            files={"file": ("patch_2ch.npy", buf_npy, "application/octet-stream")},
        )
        print(f"   Status: {res_npy.status_code}")
        print(f"   Payload size: {len(res_npy.content):,} bytes")
        assert res_npy.status_code == 200
        assert res_npy.headers.get("content-type") == "image/png"

        # 6. /thermal-preview Colormap Endpoint
        print("\n6. /thermal-preview Colormap Endpoint:")
        buf_npy.seek(0)
        res_preview = client.post(
            "/thermal-preview",
            files={"file": ("patch_2ch.npy", buf_npy, "application/octet-stream")},
        )
        print(f"   Status: {res_preview.status_code}")
        print(f"   Payload size: {len(res_preview.content):,} bytes")
        assert res_preview.status_code == 200
        assert res_preview.headers.get("content-type") == "image/png"

        # 7. /postprocess/clahe Endpoint
        print("\n7. /postprocess/clahe Contrast Enhancement Endpoint:")
        res_clahe = client.post(
            "/postprocess/clahe",
            files={"file": ("colorized.png", io.BytesIO(res_tiff.content), "image/png")},
        )
        print(f"   Status: {res_clahe.status_code}")
        print(f"   Payload size: {len(res_clahe.content):,} bytes")
        assert res_clahe.status_code == 200

    print("\n" + "=" * 80)
    print("ALL E2E API ENDPOINTS & FLOWS VERIFIED 100% OPERATIONAL")
    print("=" * 80)

if __name__ == "__main__":
    test_full_api_flow()
