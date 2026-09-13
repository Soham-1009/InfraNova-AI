"""
FastAPI backend for InfraNova AI.

Serves the Pix2Pix colorization model via a REST API.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.inference import InferenceEngine
from demo.utils import visualize_tir_as_thermal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "best" / "pix2pix_landsat_best.pth"
IMAGE_SIZE = 128

# ---------------------------------------------------------------------------
# Serve built React frontend (production Docker build)
# ---------------------------------------------------------------------------
FRONTEND_DIR = PROJECT_ROOT / "web" / "dist"


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="InfraNova AI",
    description="Thermal-to-RGB satellite image colorization powered by Pix2Pix.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Inference-Time", "X-Model"],
)

if FRONTEND_DIR.is_dir():
    from starlette.responses import FileResponse

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")


# Lazy-loaded inference engine (loaded on first request)
engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    """Get or create the inference engine singleton."""
    global engine
    if engine is None:
        engine = InferenceEngine(
            checkpoint_path=str(CHECKPOINT_PATH),
            image_size=IMAGE_SIZE,
        )
        engine.load_model()
    return engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    eng = get_engine()
    return {
        "status": "ok",
        "model_loaded": eng.model is not None,
        "device": str(eng.device),
    }


@app.post("/colorize")
@app.post("/predict")
async def colorize(
    file: UploadFile | None = None,
    band10: UploadFile | None = None,
    band11: UploadFile | None = None,
    tta: bool = False,
):
    """
    Colorize a thermal IR image or dual-band (Band 10 + Band 11) Landsat 9 observation.

    Accepts:
        - file: .tif, .tiff, .png, .jpg, .jpeg, .npy (single or 2-band raster)
        - band10 + band11: Separate upload of Band 10 and Band 11 .npy / .tif files
    Returns:
        PNG image of the colorized RGB output.
    """
    if file is None and (band10 is None or band11 is None):
        raise HTTPException(400, "Must provide either 'file' or both 'band10' and 'band11'")

    try:
        if band10 is not None and band11 is not None:
            raw_b10 = await band10.read()
            raw_b11 = await band11.read()
            if len(raw_b10) == 0 or len(raw_b11) == 0:
                raise HTTPException(400, "Uploaded band10 or band11 file is empty (0 bytes).")
            b10_arr = (
                np.load(io.BytesIO(raw_b10))
                if (band10.filename or "").endswith(".npy")
                else np.array(Image.open(io.BytesIO(raw_b10)))
            )
            b11_arr = (
                np.load(io.BytesIO(raw_b11))
                if (band11.filename or "").endswith(".npy")
                else np.array(Image.open(io.BytesIO(raw_b11)))
            )
            image_input = (b10_arr, b11_arr)
        else:
            assert file is not None
            allowed = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}
            suffix = Path(file.filename or "upload.png").suffix.lower()
            if suffix not in allowed:
                raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed}")

            raw_bytes = await file.read()
            if len(raw_bytes) == 0:
                raise HTTPException(400, "Uploaded file is empty (0 bytes).")

            if suffix == ".npy":
                image_input = np.load(io.BytesIO(raw_bytes))
            else:
                image_input = Image.open(io.BytesIO(raw_bytes))

        eng = get_engine()
        start = time.perf_counter()
        result = eng.predict(image_input, use_tta=tta)
        elapsed = time.perf_counter() - start

        # Encode result as PNG
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "X-Inference-Time": f"{elapsed:.3f}",
                "X-Model": "pix2pix-landsat-exp9-resnet",
            },
        )
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid input image: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}") from exc


@app.post("/thermal-preview")
async def thermal_preview(file: UploadFile = File(...)):
    """
    Generate a thermal colormap preview of the uploaded IR image.

    Returns: PNG image with INFERNO colormap applied.
    """
    allowed = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    try:
        raw_bytes = await file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(400, "Uploaded file is empty (0 bytes).")

        if suffix == ".npy":
            arr = np.load(io.BytesIO(raw_bytes))
            image_input = arr
        else:
            image_input = Image.open(io.BytesIO(raw_bytes))

        thermal_vis = visualize_tir_as_thermal(image_input)

        buf = io.BytesIO()
        thermal_vis.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid input image: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Preview failed: {exc}") from exc


@app.post("/postprocess/clahe")
async def apply_clahe(file: UploadFile = File(...), clip_limit: float = 2.0, grid_size: int = 8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an RGB image.

    Works in LAB color space to enhance contrast without distorting color hue.

    Args:
        file: Input PNG/JPG image.
        clip_limit: CLAHE clip limit (default 2.0).
        grid_size: CLAHE tile grid size (default 8).

    Returns: PNG image with CLAHE applied.
    """
    try:
        import cv2

        raw_bytes = await file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(400, "Uploaded file is empty (0 bytes).")

        arr = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))

        # Convert to LAB, apply CLAHE to L channel, convert back
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        result = Image.fromarray(enhanced)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid input image: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"CLAHE failed: {exc}") from exc


# Mount static frontend assets as a catch-all route at the very end
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
