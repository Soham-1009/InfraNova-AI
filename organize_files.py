"""
Organize flat Google Drive Landsat exports into per-region folders.

Earth Engine exports files as:
    <region>_SR_B2.tif
    <region>_SR_B3.tif
    <region>_SR_B4.tif
    <region>_ST_B10.tif

The patch processor expects:
    data/landsat9/input/<region>_product/*.tif
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple


BANDS = ("SR_B2", "SR_B3", "SR_B4", "ST_B10")


def parse_export_name(path: Path) -> Tuple[str, str] | None:
    stem = path.stem
    for band in BANDS:
        suffix = f"_{band}"
        if stem.endswith(suffix):
            region = stem[: -len(suffix)]
            if region:
                return region, band
    return None


def iter_tif_files(source: Path) -> Iterable[Path]:
    yield from source.glob("*.tif")
    yield from source.glob("*.tiff")


def organize_exports(source: Path, destination: Path, move: bool = False) -> Dict[str, int]:
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    destination.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, Dict[str, Path]] = {}
    ignored = 0

    for tif_path in sorted(iter_tif_files(source)):
        parsed = parse_export_name(tif_path)
        if parsed is None:
            ignored += 1
            continue
        region, band = parsed
        grouped.setdefault(region, {})[band] = tif_path

    copied = 0
    incomplete = 0

    for region, files_by_band in sorted(grouped.items()):
        missing = [band for band in BANDS if band not in files_by_band]
        if missing:
            incomplete += 1
            print(f"Skipping {region}: missing {', '.join(missing)}")
            continue

        region_dir = destination / f"{region}_product"
        region_dir.mkdir(parents=True, exist_ok=True)

        for band in BANDS:
            src = files_by_band[band]
            dst = region_dir / src.name
            if dst.exists():
                continue
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(src, dst)
            copied += 1

    return {
        "regions_found": len(grouped),
        "regions_incomplete": incomplete,
        "files_written": copied,
        "files_ignored": ignored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize Landsat 9 Earth Engine exports.")
    parser.add_argument(
        "--source",
        default="data/landsat9/downloads",
        help="Folder containing downloaded flat .tif/.tiff exports.",
    )
    parser.add_argument(
        "--destination",
        default="data/landsat9/input",
        help="Folder where per-region *_product folders will be created.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    args = parser.parse_args()

    stats = organize_exports(
        source=Path(args.source),
        destination=Path(args.destination),
        move=args.move,
    )

    print("Organization complete")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
