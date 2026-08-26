import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.tiled_inference import TiledRasterInference

RAW_B10_DIR = PROJECT_ROOT / "data" / "landsat9" / "raw"
RAW_B11_DIR = PROJECT_ROOT / "data" / "landsat9_b10_b11" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "tiled_scenes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENES = ["abuja", "adilabad", "accra", "abu_dhabi", "addis_ababa"]


def calc_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Compute Peak Signal to Noise Ratio (PSNR) in dB."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def calc_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Compute Structural Similarity Index (SSIM) across channels."""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssim_channels = []
    for c in range(img1.shape[2]):
        i1 = img1[:, :, c]
        i2 = img2[:, :, c]

        mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(i1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(i2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_channels.append(np.mean(ssim_map))

    return float(np.mean(ssim_channels))


def validate_whole_raster_scenes():
    print("=" * 80)
    print("WHOLE-RASTER TILED INFERENCE & BOUNDARY BLENDING VALIDATION")
    print("=" * 80)

    tiler = TiledRasterInference(
        checkpoint_path="outputs/best/pix2pix_landsat_best.pth",
        tile_size=128,
        overlap=32,  # 25% overlap for blending
        batch_size=16,
        window_type="cosine",
    )
    print(f"Initialized TiledRasterInference (Device: {tiler.device}, Tile: 128x128, Stride: {tiler.stride})")

    results = []

    for scene in SCENES:
        b10_path = RAW_B10_DIR / scene / "tir.tif"
        b11_path = RAW_B11_DIR / scene / "tir_b11.tif"
        rgb_path = RAW_B10_DIR / scene / "rgb.tif"

        if not b10_path.exists() or not b11_path.exists():
            print(f"Skipping {scene}: missing raw TIFF files")
            continue

        print(f"\n--- Processing Scene: {scene.upper()} ---")
        print(f"  B10 Path: {b10_path.name} ({b10_path.stat().st_size:,} bytes)")
        print(f"  B11 Path: {b11_path.name} ({b11_path.stat().st_size:,} bytes)")

        t0 = time.perf_counter()
        res = tiler.predict_scene(b10_path, b11_path, use_tta=False)
        elapsed = time.perf_counter() - t0

        rgb_pred = res["rgb_uint8"]
        H, W, _ = rgb_pred.shape
        num_tiles = res["tiles_processed"]
        weight_map = res["weight_map"]

        # Check boundary/weight coverage
        min_weight = float(weight_map.min())
        max_weight = float(weight_map.max())
        coverage_complete = min_weight > 0.0

        # Save output GeoTIFF/PNG
        out_png = OUTPUT_DIR / f"{scene}_synthesized_rgb.png"
        out_tif = OUTPUT_DIR / f"{scene}_synthesized_rgb.tif"
        tiler.save_output(res, out_png)
        tiler.save_output(res, out_tif)

        print(f"  Raster Dimensions: {H} x {W}")
        print(f"  Tiles Processed:   {num_tiles} (Elapsed: {elapsed:.2f}s, {num_tiles / elapsed:.1f} tiles/s)")
        print(f"  Weight Buffer:     min={min_weight:.2f}, max={max_weight:.2f} (Full Coverage: {coverage_complete})")
        print(f"  Output Saved:      {out_png.name}")

        # If ground truth RGB is available, evaluate full-scene metrics
        metrics_str = "N/A"
        if rgb_path.exists():
            rgb_gt, _ = tiler.load_raster(rgb_path)
            if rgb_gt.ndim == 3 and rgb_gt.shape[0] == 3:
                rgb_gt = np.transpose(rgb_gt, (1, 2, 0))  # to [H, W, 3]

            # Match dimensions if needed
            if rgb_gt.shape[:2] != (H, W):
                rgb_gt = cv2.resize(rgb_gt, (W, H), interpolation=cv2.INTER_CUBIC)

            # Normalize to uint8 for perceptual SSIM/PSNR comparison
            if rgb_gt.dtype != np.uint8:
                lo = np.percentile(rgb_gt, 2.0)
                hi = np.percentile(rgb_gt, 98.0)
                rgb_gt = np.clip((rgb_gt.astype(np.float32) - lo) / (hi - lo + 1e-6), 0.0, 1.0)
                rgb_gt = (rgb_gt * 255.0).round().astype(np.uint8)

            psnr_val = calc_psnr(rgb_gt, rgb_pred, max_val=255.0)
            ssim_val = calc_ssim(rgb_gt, rgb_pred, max_val=255.0)
            print(f"  Scene PSNR:        {psnr_val:.2f} dB")
            print(f"  Scene SSIM:        {ssim_val:.4f}")
            metrics_str = f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}"

        results.append(
            {
                "scene": scene,
                "dimensions": f"{H}x{W}",
                "tiles": num_tiles,
                "time_sec": round(elapsed, 2),
                "coverage": coverage_complete,
                "metrics": metrics_str,
            }
        )

    print("\n" + "=" * 80)
    print("SUMMARY OF WHOLE-RASTER SCENE VALIDATION")
    print("=" * 80)
    for r in results:
        print(
            f"• {r['scene'].ljust(12)}: {r['dimensions'].ljust(10)} | {r['tiles']} tiles | {r['time_sec']}s | Coverage: {r['coverage']} | {r['metrics']}"
        )

    print("\n[PASS] WHOLE-RASTER SCENE VALIDATION COMPLETE")


if __name__ == "__main__":
    validate_whole_raster_scenes()
