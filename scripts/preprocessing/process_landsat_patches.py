"""
Process raw Landsat 9 rgb.tif / tir.tif files into aligned training patches.

Each region folder is expected to contain exactly two files:
    rgb.tif   - 3-band surface reflectance (Red, Green, Blue)
    tir.tif   - 1-band thermal infrared

Output structure:
    <output_dir>/<region>/sample_000/
        tir_200m.npy   (64 x 64)
        tir_100m.npy   (128 x 128)
        rgb_100m.npy   (3 x 128 x 128)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATCH_SIZE_200M = 64    # Input patch size at 200 m resolution
PATCH_SIZE_100M = 128   # Output patch size at 100 m resolution (2x)
STRIDE = 16             # Stride in 200 m pixels (75 % overlap)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
# I/O helpers
# ===================================================================

def load_rgb(path: Path) -> np.ndarray:
    """Load an RGB GeoTIFF and return a (3, H, W) float32 array."""
    img = tifffile.imread(path).astype(np.float32)

    if img.ndim == 2:
        raise ValueError(f"Expected 3-band RGB, got single-band image: {path}")

    if img.ndim == 3:
        # (H, W, 3) -> (3, H, W)
        if img.shape[2] in (3, 4):
            img = np.transpose(img[:, :, :3], (2, 0, 1))
        # (3, H, W) already
        elif img.shape[0] in (3, 4):
            img = img[:3]
        else:
            raise ValueError(
                f"Ambiguous RGB shape {img.shape} in {path}. "
                "Expected (H, W, 3) or (3, H, W)."
            )

    if img.shape[0] != 3:
        raise ValueError(f"RGB must have 3 bands, got {img.shape[0]}: {path}")

    return img


def load_tir(path: Path) -> np.ndarray:
    """Load a thermal GeoTIFF and return a (H, W) float32 array."""
    img = tifffile.imread(path).astype(np.float32)

    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            raise ValueError(
                f"Expected single-band TIR, got shape {img.shape}: {path}"
            )

    if img.ndim != 2:
        raise ValueError(f"TIR must be 2-D, got ndim={img.ndim}: {path}")

    return img


# ===================================================================
# Resampling
# ===================================================================

def resize_2d(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a 2-D array using INTER_AREA (anti-aliased downsampling)."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_bands(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a (C, H, W) array band-by-band."""
    bands = [resize_2d(band, height, width) for band in image]
    return np.stack(bands)


# ===================================================================
# Core processing
# ===================================================================

def process_region(region_dir: Path, output_dir: Path) -> int:
    """
    Process one region folder into aligned patches.

    Returns the number of patches created (0 on skip / error).
    """
    region_id = region_dir.name
    rgb_path = region_dir / "rgb.tif"
    tir_path = region_dir / "tir.tif"

    # ------ existence check ------
    if not rgb_path.exists() or not tir_path.exists():
        logger.warning("Skipping %s: missing rgb.tif or tir.tif", region_id)
        return 0

    # ------ load ------
    try:
        rgb = load_rgb(rgb_path)
        tir = load_tir(tir_path)
    except Exception as exc:
        logger.warning("Skipping %s: %s", region_id, exc)
        return 0

    # ------ finite-value check ------
    if not np.isfinite(tir).all():
        logger.warning("Skipping %s: TIR contains non-finite values", region_id)
        return 0
    if not np.isfinite(rgb).all():
        logger.warning("Skipping %s: RGB contains non-finite values", region_id)
        return 0

    # ------ dimension compatibility ------
    rgb_hw = rgb.shape[1:]          # (H, W) from (3, H, W)
    tir_hw = tir.shape              # (H, W)

    if rgb_hw != tir_hw:
        logger.warning(
            "Skipping %s: RGB spatial %s != TIR spatial %s",
            region_id, rgb_hw, tir_hw,
        )
        return 0

    # ------ compute target grids ------
    # Landsat 9 native is 30 m.  We want 200 m and 100 m grids.
    # The 200 m grid is derived first; the 100 m grid is exactly 2x.
    source_h, source_w = tir_hw
    h200 = max(1, round(source_h * 3.0 / 20.0))
    w200 = max(1, round(source_w * 3.0 / 20.0))
    h100, w100 = h200 * 2, w200 * 2

    # ------ resample ------
    tir_200m = resize_2d(tir, h200, w200)
    tir_100m = resize_2d(tir, h100, w100)
    rgb_100m = resize_bands(rgb, h100, w100)

    logger.info(
        "Processing %s  |  source %dx%d  ->  200 m %dx%d  /  100 m %dx%d",
        region_id, source_w, source_h, w200, h200, w100, h100,
    )

    # ------ patch feasibility ------
    if h200 < PATCH_SIZE_200M or w200 < PATCH_SIZE_200M:
        logger.warning(
            "Skipping %s: 200 m grid %dx%d too small for %dx%d patches",
            region_id, w200, h200, PATCH_SIZE_200M, PATCH_SIZE_200M,
        )
        return 0

    # ------ prepare output directory ------
    region_out = output_dir / region_id
    if region_out.exists():
        shutil.rmtree(region_out)
    region_out.mkdir(parents=True, exist_ok=True)

    # ------ extract patches ------
    count = 0
    skipped_border = 0

    for y in range(0, h200 - PATCH_SIZE_200M + 1, STRIDE):
        for x in range(0, w200 - PATCH_SIZE_200M + 1, STRIDE):
            # 200 m patch
            p_tir200 = tir_200m[y:y + PATCH_SIZE_200M,
                                x:x + PATCH_SIZE_200M]

            # Corresponding 100 m patch (exact 2x alignment)
            y1, x1 = y * 2, x * 2
            p_tir100 = tir_100m[y1:y1 + PATCH_SIZE_100M,
                                x1:x1 + PATCH_SIZE_100M]
            p_rgb100 = rgb_100m[:, y1:y1 + PATCH_SIZE_100M,
                                   x1:x1 + PATCH_SIZE_100M]

            # Guard against rounding-induced border slivers
            if p_tir100.shape != (PATCH_SIZE_100M, PATCH_SIZE_100M):
                skipped_border += 1
                continue
            if p_rgb100.shape[1:] != (PATCH_SIZE_100M, PATCH_SIZE_100M):
                skipped_border += 1
                continue

            # Save
            patch_dir = region_out / f"sample_{count:03d}"
            patch_dir.mkdir(exist_ok=True)

            np.save(patch_dir / "tir_200m.npy", p_tir200)
            np.save(patch_dir / "tir_100m.npy", p_tir100)
            np.save(patch_dir / "rgb_100m.npy", p_rgb100)

            count += 1

    if skipped_border > 0:
        logger.info("  %s: skipped %d border patches", region_id, skipped_border)
    logger.info("  %s: created %d patches", region_id, count)

    return count


# ===================================================================
# CLI entry point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process raw Landsat 9 rgb/tir TIFFs into training patches.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/landsat9/raw",
        help="Directory containing region folders with rgb.tif and tir.tif",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/landsat9/patches",
        help="Directory to write extracted patches",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root if they are not absolute
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    regions = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not regions:
        logger.error("No region folders found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d regions in %s", len(regions), input_dir)
    logger.info("Output -> %s", output_dir)
    logger.info(
        "Patch config: 200 m = %dx%d, 100 m = %dx%d, stride = %d",
        PATCH_SIZE_200M, PATCH_SIZE_200M,
        PATCH_SIZE_100M, PATCH_SIZE_100M,
        STRIDE,
    )

    t0 = time.perf_counter()
    total_patches = 0
    processed = 0
    skipped = 0
    patch_counts = []

    for i, region_dir in enumerate(regions, 1):
        try:
            n = process_region(region_dir, output_dir)
            if n > 0:
                total_patches += n
                processed += 1
                patch_counts.append(n)
            else:
                skipped += 1
        except Exception as exc:
            logger.error("Region %s failed: %s", region_dir.name, exc)
            skipped += 1

        if i % 50 == 0:
            logger.info("Progress: %d / %d regions", i, len(regions))

    elapsed = time.perf_counter() - t0

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  Regions processed : %d", processed)
    logger.info("  Regions skipped   : %d", skipped)
    logger.info("  Total patches     : %d", total_patches)
    if processed > 0:
        avg_patches = total_patches / processed
        min_patches = min(patch_counts) if patch_counts else 0
        max_patches = max(patch_counts) if patch_counts else 0
        logger.info("  Avg patches/region: %.1f", avg_patches)
        logger.info("  Min patches/region: %d", min_patches)
        logger.info("  Max patches/region: %d", max_patches)
    logger.info("  Elapsed           : %.1f s", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
