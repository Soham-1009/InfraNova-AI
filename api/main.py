"""
FastAPI backend for InfraNova AI.

Serves the Pix2Pix colorization model via a REST API.
"""

from __future__ import annotations

import io
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
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
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "models" / "best" / "pix2pix_landsat_best.pth"
IMAGE_SIZE = 128

# ---------------------------------------------------------------------------
# Serve built React frontend (production Docker build)
# ---------------------------------------------------------------------------
FRONTEND_DIR = PROJECT_ROOT / "web" / "dist"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Mount static frontend assets on startup if a production build exists."""
    if FRONTEND_DIR.is_dir():
        application.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
    yield


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="InfraNova AI",
    description="Thermal-to-RGB satellite image colorization powered by Pix2Pix.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/colorize")
async def colorize(file: UploadFile = File(...), tta: bool = False):
    """
    Colorize a thermal IR image.

    Accepts: .tif, .tiff, .png, .jpg, .jpeg, .npy
    Returns: PNG image of the colorized RGB output.
    """
    allowed = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed}")

    try:
        raw_bytes = await file.read()

        if suffix == ".npy":
            arr = np.load(io.BytesIO(raw_bytes))
            image_input = arr
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
                "X-Model": "pix2pix-landsat-epoch226",
            },
        )
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
    except Exception as exc:
        raise HTTPException(500, f"CLAHE failed: {exc}") from exc
