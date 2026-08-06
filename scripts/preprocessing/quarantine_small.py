"""
Quarantine small Landsat regions that fail the 64x64 minimum grid check,
using rasterio to avoid cv2 import errors.
"""

import shutil
from pathlib import Path

import rasterio

# Make project root importable
PROJECT_ROOT = Path("c:/Users/soham/Desktop/Soham/InfraNova-AI")

def main():
    raw_dir = PROJECT_ROOT / "data/landsat9/raw"
    quarantine_dir = PROJECT_ROOT / "data/landsat9/quarantine"

    if not raw_dir.exists():
        print(f"Raw directory not found: {raw_dir}")
        return

    quarantine_dir.mkdir(parents=True, exist_ok=True)

    quarantined = 0
    total = 0

    # Write to a file since stdout is swallowed
    with open(PROJECT_ROOT / "quarantine_log.txt", "w") as log:
        log.write(f"Scanning {raw_dir} for undersized regions...\n")

        for region_dir in sorted(raw_dir.iterdir()):
            if not region_dir.is_dir():
                continue

            total += 1
            rgb_path = region_dir / "rgb.tif"
            tir_path = region_dir / "tir.tif"

            if not rgb_path.exists() or not tir_path.exists():
                continue

            try:
                with rasterio.open(tir_path) as src:
                    source_w, source_h = src.width, src.height
            except Exception as e:
                log.write(f"Error loading {region_dir.name}: {e}\n")
                continue

            # EXACT logic from process_landsat_patches.py
            h200 = max(1, round(source_h * 3.0 / 20.0))
            w200 = max(1, round(source_w * 3.0 / 20.0))

            if h200 < 64 or w200 < 64:
                log.write(f"Quarantining {region_dir.name:<25} | source {source_w}x{source_h} -> 200m grid {w200}x{h200}\n")
                shutil.move(str(region_dir), str(quarantine_dir / region_dir.name))
                quarantined += 1

        log.write("-" * 50 + "\n")
        log.write(f"Total checked: {total}\n")
        log.write(f"Quarantined  : {quarantined}\n")

if __name__ == "__main__":
    main()
