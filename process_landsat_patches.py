"""
Process raw Landsat 9 TIF files into training patches.
"""

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# Patch size configuration
PATCH_SIZE_200M = 64   # Input patch size at 200m
PATCH_SIZE_100M = 128  # Output patch size at 100m (2x)
STRIDE = 16            # 75% overlap


def resize_image(image, height, width):
    if height <= 0 or width <= 0:
        raise ValueError("Target image dimensions must be positive")

    if image.ndim == 2:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    bands = []
    for band in image:
        bands.append(cv2.resize(band, (width, height), interpolation=cv2.INTER_AREA))
    return np.stack(bands)


def merge_rgb(b2_path, b3_path, b4_path):
    blue = tifffile.imread(b2_path)
    green = tifffile.imread(b3_path)
    red = tifffile.imread(b4_path)
    return np.stack([red, green, blue], axis=0)


def process_region(region_dir, output_dir):
    region_id = region_dir.name.replace('_product', '')
    print(f"\nProcessing {region_id}...")
    
    files = sorted(region_dir.glob("*.tif"))
    b2 = next((f for f in files if 'SR_B2' in f.name), None)
    b3 = next((f for f in files if 'SR_B3' in f.name), None)
    b4 = next((f for f in files if 'SR_B4' in f.name), None)
    b10 = next((f for f in files if 'ST_B10' in f.name), None)
    
    if not all([b2, b3, b4, b10]):
        print(f"  Missing bands")
        return 0
    
    try:
        tir = tifffile.imread(b10)
        rgb = merge_rgb(b2, b3, b4)
    except Exception as exc:
        logger.warning("  Skipping %s: corrupted TIFF (%s)", region_id, exc)
        return 0
    
    # Check for non-finite values in raw data
    if not np.isfinite(tir).all():
        logger.warning("  Skipping %s: TIR contains non-finite values", region_id)
        return 0
    if not np.isfinite(rgb).all():
        logger.warning("  Skipping %s: RGB contains non-finite values", region_id)
        return 0

    if rgb.shape[-2:] != tir.shape:
        logger.warning(
            "  Skipping %s: RGB (%s) and TIR (%s) source grids differ",
            region_id,
            rgb.shape[-2:],
            tir.shape,
        )
        return 0

    # Derive both resolutions from one integer 200 m grid so each 200 m
    # patch maps to exactly twice as many 100 m pixels without border drift.
    source_h, source_w = tir.shape
    h200 = max(1, round(source_h * 3.0 / 20.0))
    w200 = max(1, round(source_w * 3.0 / 20.0))
    h100, w100 = h200 * 2, w200 * 2

    tir_200m = resize_image(tir, h200, w200)
    tir_100m = resize_image(tir, h100, w100)
    rgb_100m = resize_image(rgb, h100, w100)
    
    h200, w200 = tir_200m.shape
    print(f"  200m TIR size: {h200}x{w200}")
    print(f"  Using patch size: {PATCH_SIZE_200M}x{PATCH_SIZE_200M}, stride: {STRIDE}")
    
    if h200 < PATCH_SIZE_200M or w200 < PATCH_SIZE_200M:
        print(f"  Image too small for {PATCH_SIZE_200M}x{PATCH_SIZE_200M} patches")
        return 0
    
    count = 0
    skipped_border = 0
    region_patches_dir = output_dir / region_id
    if region_patches_dir.exists():
        shutil.rmtree(region_patches_dir)
    region_patches_dir.mkdir(parents=True, exist_ok=True)
    
    for y in range(0, h200 - PATCH_SIZE_200M + 1, STRIDE):
        for x in range(0, w200 - PATCH_SIZE_200M + 1, STRIDE):
            patch_tir_200m = tir_200m[y:y+PATCH_SIZE_200M, x:x+PATCH_SIZE_200M]
            
            y100, x100 = y*2, x*2
            patch_tir_100m = tir_100m[y100:y100+PATCH_SIZE_100M, x100:x100+PATCH_SIZE_100M]
            patch_rgb_100m = rgb_100m[:, y100:y100+PATCH_SIZE_100M, x100:x100+PATCH_SIZE_100M]
            
            if patch_tir_100m.shape != (PATCH_SIZE_100M, PATCH_SIZE_100M):
                skipped_border += 1
                continue
            if patch_rgb_100m.shape[-2:] != (PATCH_SIZE_100M, PATCH_SIZE_100M):
                skipped_border += 1
                continue
            
            patch_dir = region_patches_dir / f'sample_{count:03d}'
            patch_dir.mkdir(exist_ok=True)
            
            np.save(patch_dir / 'tir_200m.npy', patch_tir_200m)
            np.save(patch_dir / 'tir_100m.npy', patch_tir_100m)
            np.save(patch_dir / 'rgb_100m.npy', patch_rgb_100m)
            
            count += 1
    
    if skipped_border > 0:
        print(f"  Skipped {skipped_border} border patches (misaligned 200m/100m grids)")
    print(f"  Created {count} patches")
    return count


def main():
    input_dir = Path("data/landsat9/input")
    output_dir = Path("data/landsat9/patches")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("Download and organize Landsat exports before processing patches.")
        return
    
    regions = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not regions:
        print(f"No regions found in {input_dir}")
        return
    
    total = 0
    for region_dir in regions:
        total += process_region(region_dir, output_dir)
    
    print(f"\n{'='*50}")
    print(f"Total patches: {total}")
    print(f"Patch sizes: {PATCH_SIZE_200M}x{PATCH_SIZE_200M} (input), {PATCH_SIZE_100M}x{PATCH_SIZE_100M} (output)")


if __name__ == '__main__':
    main()
