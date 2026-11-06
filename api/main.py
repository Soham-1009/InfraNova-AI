"""
FastAPI backend for InfraNova AI.

Serves the Pix2Pix colorization model via a REST API.
"""

from __future__ import annotations

import io
import logging
import math
import os
import sys
import time
from pathlib import Path
from threading import RLock

import numpy as np
import tifffile
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.inference import InferenceEngine
from demo.utils import to_single_band_array, visualize_tir_as_thermal
from src.inference.tiled_inference import TiledRasterInference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Security Limits
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path(
    os.getenv(
        "INFRANOVA_CHECKPOINT",
        os.getenv("CHECKPOINT_PATH", str(PROJECT_ROOT / "outputs" / "best" / "pix2pix_landsat_best.pth")),
    )
)
DEVICE = os.getenv("INFRANOVA_DEVICE", os.getenv("DEVICE", None))
IMAGE_SIZE = int(os.getenv("INFRANOVA_IMAGE_SIZE", "128"))
CORS_ORIGINS_RAW = os.getenv("INFRANOVA_CORS_ORIGINS", "*")
CORS_ORIGINS = [orig.strip() for orig in CORS_ORIGINS_RAW.split(",") if orig.strip()]
if not CORS_ORIGINS:
    CORS_ORIGINS = ["*"]

# DoS, decompression bomb, and upload size safeguards
Image.MAX_IMAGE_PIXELS = 25_000_000
MAX_UPLOAD_SIZE_MB = int(os.getenv("INFRANOVA_MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_SIZE = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_IMAGE_DIM = int(os.getenv("INFRANOVA_MAX_IMAGE_DIM", "4096"))
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}

# ---------------------------------------------------------------------------
# Serve built React frontend (production Docker build)
# ---------------------------------------------------------------------------
FRONTEND_DIR = PROJECT_ROOT / "web" / "dist"


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
docs_url: str | None = os.getenv("INFRANOVA_DOCS_URL", "/docs")
redoc_url: str | None = os.getenv("INFRANOVA_REDOC_URL", "/redoc")
openapi_url: str | None = os.getenv("INFRANOVA_OPENAPI_URL", "/openapi.json")
if os.getenv("INFRANOVA_ENABLE_DOCS", "true").lower() in ("false", "0", "no"):
    docs_url = None
    redoc_url = None
    openapi_url = None

app = FastAPI(
    title="InfraNova AI",
    description="Thermal-to-RGB satellite image colorization powered by Pix2Pix.",
    version="1.0.0",
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    """Add defense-in-depth HTTP security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'"
    )
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=()"
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=not ("*" in CORS_ORIGINS or CORS_ORIGINS == ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Inference-Time", "X-Model", "X-Spatial-Grid", "X-Tiles-Processed"],
)


async def read_upload_file(upload_file: UploadFile, max_size: int | None = None) -> bytes:
    """Read uploaded file in chunks with strict size limits to prevent memory exhaustion DoS."""
    effective_max = MAX_UPLOAD_SIZE if max_size is None else max_size
    chunks: list[bytes] = []
    total_size = 0
    chunk_size = 64 * 1024
    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > effective_max:
            raise HTTPException(
                413,
                f"Uploaded file exceeds maximum permitted size of {effective_max // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if len(data) == 0:
        raise HTTPException(400, "Uploaded file is empty (0 bytes).")
    return data


def _validate_tiff_dimensions(raw_bytes: bytes, max_dim: int = MAX_IMAGE_DIM) -> None:
    """Pre-flight check TIFF IFDs to prevent allocating memory for decompression bombs."""
    try:
        with tifffile.TiffFile(io.BytesIO(raw_bytes)) as tif:
            for series in tif.series:
                shape = series.shape
                if len(shape) >= 2:
                    h, w = (
                        (shape[0], shape[1])
                        if len(shape) == 2
                        else ((shape[1], shape[2]) if shape[0] in (1, 2, 3, 4) else (shape[0], shape[1]))
                    )
                    if h > max_dim or w > max_dim:
                        raise HTTPException(
                            400,
                            f"Image dimensions ({w}x{h}) exceed maximum permitted dimension ({max_dim}x{max_dim}).",
                        )
    except HTTPException:
        raise
    except Exception:
        pass


# Lazy-loaded inference engines (loaded on first request)
engine: InferenceEngine | None = None
_engine_lock = RLock()
tiled_engine: TiledRasterInference | None = None
_tiled_lock = RLock()


def get_engine() -> InferenceEngine:
    """Get or create the inference engine singleton."""
    global engine
    if engine is not None:
        return engine

    with _engine_lock:
        if engine is None:
            candidate = InferenceEngine(
                checkpoint_path=str(CHECKPOINT_PATH),
                device=DEVICE,
                image_size=IMAGE_SIZE,
            )
            candidate.load_model()
            engine = candidate
    return engine


def get_tiled_engine() -> TiledRasterInference:
    """Get or create the tiled raster inference engine singleton."""
    global tiled_engine
    if tiled_engine is not None:
        return tiled_engine

    with _tiled_lock:
        if tiled_engine is None:
            eng = get_engine()
            tiled_engine = TiledRasterInference(
                checkpoint_path=str(CHECKPOINT_PATH),
                tile_size=IMAGE_SIZE,
                device=DEVICE,
                engine=eng,
            )
    return tiled_engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        eng = get_engine()
        return {
            "status": "ok",
            "model_loaded": eng.model is not None,
            "device": str(eng.device),
        }
    except (FileNotFoundError, OSError):
        return {
            "status": "degraded",
            "model_loaded": False,
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
        image_input: Image.Image | np.ndarray | tuple[np.ndarray, np.ndarray]
        if band10 is not None and band11 is not None:
            b10_fn = Path(band10.filename or "").name.lower()
            b11_fn = Path(band11.filename or "").name.lower()
            b10_suffix = Path(b10_fn).suffix
            b11_suffix = Path(b11_fn).suffix
            if b10_suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(400, f"Unsupported file type for band10: '{b10_suffix}'. Allowed: {ALLOWED_EXTENSIONS}")
            if b11_suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(400, f"Unsupported file type for band11: '{b11_suffix}'. Allowed: {ALLOWED_EXTENSIONS}")

            raw_b10 = await read_upload_file(band10)
            raw_b11 = await read_upload_file(band11)

            if b10_suffix == ".npy":
                b10_arr = np.load(io.BytesIO(raw_b10), allow_pickle=False)
            elif b10_suffix in {".tif", ".tiff"}:
                _validate_tiff_dimensions(raw_b10)
                b10_arr = tifffile.imread(io.BytesIO(raw_b10))
            else:
                b10_arr = np.array(Image.open(io.BytesIO(raw_b10)))

            if b11_suffix == ".npy":
                b11_arr = np.load(io.BytesIO(raw_b11), allow_pickle=False)
            elif b11_suffix in {".tif", ".tiff"}:
                _validate_tiff_dimensions(raw_b11)
                b11_arr = tifffile.imread(io.BytesIO(raw_b11))
            else:
                b11_arr = np.array(Image.open(io.BytesIO(raw_b11)))

            image_input = (b10_arr, b11_arr)
        else:
            assert file is not None
            suffix = Path(file.filename or "upload.png").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}")

            raw_bytes = await read_upload_file(file)

            if suffix == ".npy":
                image_input = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
            elif suffix in {".tif", ".tiff"}:
                _validate_tiff_dimensions(raw_bytes)
                image_input = tifffile.imread(io.BytesIO(raw_bytes))
            else:
                image_input = Image.open(io.BytesIO(raw_bytes))

        eng = get_engine()
        start = time.perf_counter()

        # Check spatial dimensions to decide between tiled vs single-patch inference
        b10_arr = None
        b11_arr = None
        H, W = IMAGE_SIZE, IMAGE_SIZE

        if isinstance(image_input, (tuple, list)) and len(image_input) == 2:
            b10_arr = to_single_band_array(image_input[0])
            b11_arr = to_single_band_array(image_input[1])
            if b10_arr.shape != b11_arr.shape:
                raise HTTPException(
                    400,
                    f"Band 10 spatial shape {b10_arr.shape} does not match Band 11 shape {b11_arr.shape}.",
                )
            H, W = b10_arr.shape
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                H, W = image_input.shape
                b10_arr = b11_arr = image_input
            elif image_input.ndim == 3 and image_input.shape[0] >= 2 and (image_input.shape[0] < image_input.shape[1] or image_input.shape[2] > 4):
                b10_arr, b11_arr = image_input[0], image_input[1]
                H, W = b10_arr.shape
            elif image_input.ndim == 3 and image_input.shape[2] >= 2:
                b10_arr, b11_arr = image_input[:, :, 0], image_input[:, :, 1]
                H, W = b10_arr.shape
            else:
                b10_arr = to_single_band_array(image_input)
                b11_arr = b10_arr
                H, W = b10_arr.shape
        elif isinstance(image_input, Image.Image):
            W, H = image_input.size
            b10_arr = to_single_band_array(np.array(image_input))
            b11_arr = b10_arr

        if H <= 0 or W <= 0:
            raise HTTPException(400, "Image has invalid spatial dimensions (width and height must be > 0).")

        if H > MAX_IMAGE_DIM or W > MAX_IMAGE_DIM:
            raise HTTPException(
                400,
                f"Image dimensions ({W}x{H}) exceed maximum permitted dimension ({MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}).",
            )

        tiles_processed = 1
        if (H > IMAGE_SIZE or W > IMAGE_SIZE) and b10_arr is not None and b11_arr is not None:
            tiler = get_tiled_engine()
            tiled_out = tiler.predict_scene(b10_arr, b11_arr, use_tta=tta)
            result = Image.fromarray(tiled_out["rgb_uint8"])
            tiles_processed = tiled_out.get("tiles_processed", 1)
        else:
            inference_input = (b10_arr, b11_arr) if (b10_arr is not None and b11_arr is not None) else image_input
            result = eng.predict(inference_input, use_tta=tta)

        elapsed = time.perf_counter() - start

        # Encode result as PNG
        buf = io.BytesIO()
        result.save(buf, format="PNG")

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Inference-Time": f"{elapsed:.3f}",
                "X-Model": "pix2pix-landsat-exp9-resnet",
                "X-Spatial-Grid": f"{W}x{H}",
                "X-Tiles-Processed": str(tiles_processed),
            },
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.error("Model checkpoint unavailable: %s", exc)
        raise HTTPException(503, "Model checkpoint unavailable.") from exc
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(400, f"Invalid input image: {exc}") from exc
    except Exception as exc:
        logger.exception("Inference processing error")
        raise HTTPException(500, "Inference failed due to an internal server error.") from exc


@app.post("/thermal-preview")
async def thermal_preview(file: UploadFile = File(...)):
    """
    Generate a thermal colormap preview of the uploaded IR image.

    Returns: PNG image with INFERNO colormap applied.
    """
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    try:
        raw_bytes = await read_upload_file(file)

        if suffix == ".npy":
            arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
            image_input = arr
        elif suffix in {".tif", ".tiff"}:
            _validate_tiff_dimensions(raw_bytes)
            image_input = tifffile.imread(io.BytesIO(raw_bytes))
        else:
            image_input = Image.open(io.BytesIO(raw_bytes))

        # Dimension checks
        if isinstance(image_input, Image.Image):
            W, H = image_input.size
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim < 2:
                raise HTTPException(400, "Input array must be at least 2-dimensional.")
            H, W = image_input.shape[:2]
        else:
            H, W = IMAGE_SIZE, IMAGE_SIZE

        if H <= 0 or W <= 0:
            raise HTTPException(400, "Image has invalid spatial dimensions (width and height must be > 0).")

        if H > MAX_IMAGE_DIM or W > MAX_IMAGE_DIM:
            raise HTTPException(
                400,
                f"Image dimensions ({W}x{H}) exceed maximum permitted dimension ({MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}).",
            )

        thermal_vis = visualize_tir_as_thermal(image_input)

        buf = io.BytesIO()
        thermal_vis.save(buf, format="PNG")

        return Response(content=buf.getvalue(), media_type="image/png")
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(400, f"Invalid input image: {exc}") from exc
    except Exception as exc:
        logger.exception("Preview generation error")
        raise HTTPException(500, "Preview failed due to an internal server error.") from exc


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

        if not math.isfinite(clip_limit) or clip_limit < 0.0 or clip_limit > 100.0:
            raise HTTPException(400, "clip_limit must be a finite value between 0.0 and 100.0.")
        if grid_size <= 0 or grid_size > 128:
            raise HTTPException(400, "grid_size must be an integer between 1 and 128.")

        suffix = Path(file.filename or "upload.png").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            raise HTTPException(
                400,
                f"Unsupported file type for CLAHE postprocessing: '{suffix}'. Allowed: .png, .jpg, .jpeg, .tif, .tiff",
            )

        raw_bytes = await read_upload_file(file)
        if suffix in {".tif", ".tiff"}:
            _validate_tiff_dimensions(raw_bytes)

        arr = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
        H, W = arr.shape[:2]
        if H <= 0 or W <= 0:
            raise HTTPException(400, "Image has invalid spatial dimensions (width and height must be > 0).")
        if H > MAX_IMAGE_DIM or W > MAX_IMAGE_DIM:
            raise HTTPException(
                400,
                f"Image dimensions ({W}x{H}) exceed maximum permitted dimension ({MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}).",
            )

        # Convert to LAB, apply CLAHE to L channel, convert back
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        result = Image.fromarray(enhanced)
        buf = io.BytesIO()
        result.save(buf, format="PNG")

        return Response(content=buf.getvalue(), media_type="image/png")
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(400, f"Invalid input image: {exc}") from exc
    except Exception as exc:
        logger.exception("CLAHE postprocessing error")
        raise HTTPException(500, "CLAHE failed due to an internal server error.") from exc


# Mount static frontend assets as a catch-all route at the very end
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
