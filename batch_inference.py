"""
Batch inference for InfraNova AI.

Process a folder (or single file) of thermal images and generate RGB outputs.

Usage:
    python batch_inference.py --input-dir path/to/thermal --output-dir path/to/output
    python batch_inference.py --input-dir single_image.tif --output-dir output/
    python batch_inference.py --input-dir path/to/thermal --output-dir output/ --recursive --tta --workers 4
    python batch_inference.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.inference import InferenceEngine

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def discover_images(input_path: Path, recursive: bool = False) -> List[Path]:
    """Find all supported image files."""
    # Single file mode
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        return []

    # Directory mode
    files = []
    pattern = "**/*" if recursive else "*"
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_path.glob(f"{pattern}{ext}"))
        files.extend(input_path.glob(f"{pattern}{ext.upper()}"))
    return sorted(set(files))


def process_single_image(
    engine: InferenceEngine,
    image_path: Path,
    output_dir: Path,
    tta: bool,
    output_format: str,
) -> Tuple[bool, str, float]:
    """
    Process one image. Returns (success, message, time_seconds).
    """
    try:
        with Image.open(image_path) as source:
            image = source.copy()

        # Convert palette to grayscale
        if image.mode == "P":
            image = image.convert("L")

        if image.mode in ("RGB", "RGBA"):
            return False, f"Skipped {image_path.name}: RGB image", 0.0

        start = time.perf_counter()
        result = engine.predict(image, tta=tta)
        elapsed = time.perf_counter() - start

        stem = image_path.stem

        if output_format in ("png", "both"):
            result.save(output_dir / f"{stem}_rgb.png")

        if output_format in ("tiff", "both"):
            arr = np.array(result)
            try:
                import rasterio
                from rasterio.transform import from_bounds

                # Preserve GeoTIFF metadata if source is GeoTIFF
                try:
                    with rasterio.open(str(image_path)) as src:
                        profile = src.profile.copy()
                        profile.update(
                            count=3, dtype="uint8",
                            driver="GTiff",
                            width=arr.shape[1],
                            height=arr.shape[0],
                            transform=from_bounds(*src.bounds, arr.shape[1], arr.shape[0]),
                        )
                        out_path = output_dir / f"{stem}_rgb.tif"
                        with rasterio.open(str(out_path), "w", **profile) as dst:
                            for band_idx in range(3):
                                dst.write(arr[:, :, band_idx], band_idx + 1)
                except Exception:
                    # Fallback: plain TIFF
                    try:
                        import tifffile
                        tifffile.imwrite(str(output_dir / f"{stem}_rgb.tif"), arr)
                    except ImportError:
                        result.save(output_dir / f"{stem}_rgb.tiff")
            except ImportError:
                try:
                    import tifffile
                    tifffile.imwrite(str(output_dir / f"{stem}_rgb.tif"), arr)
                except ImportError:
                    result.save(output_dir / f"{stem}_rgb.tiff")

        return True, f"{image_path.name}", elapsed

    except Exception as exc:
        return False, f"Failed {image_path.name}: {exc}", 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference: convert thermal images to RGB."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory or single file containing thermal input images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where generated RGB images will be saved.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best/pix2pix_landsat_best.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (4x geometric averaging).",
    )
    parser.add_argument(
        "--format",
        choices=["png", "tiff", "both"],
        default="png",
        help="Output format (default: png).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directory for images.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for inference (default: 1).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(input_path, recursive=args.recursive)
    if not image_paths:
        print(f"No supported images found in {input_path}")
        print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")
    print(f"Checkpoint: {checkpoint}")
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    print("Loading model...", end=" ", flush=True)
    try:
        engine = InferenceEngine(str(checkpoint))
    except Exception as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
    print("done.")

    total_time = 0.0
    success = 0
    failed = 0

    if args.workers <= 1:
        # Sequential mode
        try:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="Processing", unit="img")
        except ImportError:
            iterator = image_paths

        for image_path in iterator:
            ok, msg, elapsed = process_single_image(
                engine, image_path, output_dir, args.tta, args.format
            )
            if ok:
                success += 1
                total_time += elapsed
            else:
                failed += 1
                print(f"  {msg}")
    else:
        # Parallel mode
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_single_image, engine, p, output_dir, args.tta, args.format
                ): p
                for p in image_paths
            }
            for future in as_completed(futures):
                ok, msg, elapsed = future.result()
                if ok:
                    success += 1
                    total_time += elapsed
                else:
                    failed += 1
                    print(f"  {msg}")
                print(f"  [{success + failed}/{len(image_paths)}] {msg}")

    # Summary
    print("=" * 60)
    print(f"Processed: {success}/{len(image_paths)} images")
    if failed > 0:
        print(f"Failed:    {failed}")
    if success > 0:
        avg_time = total_time / success
        print(f"Total time: {total_time:.2f}s")
        print(f"Average:    {avg_time:.2f}s per image")
    print(f"Output:    {output_dir}")


if __name__ == "__main__":
    main()
