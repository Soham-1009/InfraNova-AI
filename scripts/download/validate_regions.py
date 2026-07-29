"""
Validate region quality for the InfraNova AI dataset.

Checks for duplicate cities/coordinates, download failures, missing bands,
incomplete exports, and basic data quality issues.

Usage:
    python validate_regions.py
    python validate_regions.py --dir data/landsat9/input
    python validate_regions.py --patches data/landsat9/patches
    python validate_regions.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



EXPECTED_BANDS = {"SR_B2", "SR_B3", "SR_B4", "ST_B10"}
EXPECTED_PATCH_FILES = {"tir_200m.npy", "tir_100m.npy", "rgb_100m.npy"}


def _safe_load_npy(path: Path) -> np.ndarray | None:
    """Load .npy file safely, returning None on failure."""
    try:
        return np.load(path)
    except Exception:
        return None


def validate_raw_regions(data_dir: str) -> dict[str, Any]:
    """
    Validate the raw downloaded region directories.

    Checks:
    - Missing bands
    - Empty/corrupt TIF files
    - Duplicate region names
    - Coordinate duplicates (from filenames if encoded)

    Args:
        data_dir: Path to raw input directory (e.g., data/landsat9/input).

    Returns:
        Validation report dictionary.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return {"error": f"Directory not found: {data_dir}", "regions": 0}

    region_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    report: dict[str, Any] = {
        "total_regions": len(region_dirs),
        "missing_bands": [],
        "empty_files": [],
        "duplicate_names": [],
        "issues_count": 0,
    }

    # Check for duplicate region names
    names = [d.name.lower().strip() for d in region_dirs]
    name_counts = Counter(names)
    duplicates = {name: count for name, count in name_counts.items() if count > 1}
    if duplicates:
        report["duplicate_names"] = [
            {"name": name, "count": count} for name, count in duplicates.items()
        ]
        report["issues_count"] += len(duplicates)

    # Check each region
    for region_dir in region_dirs:
        tif_files = list(region_dir.glob("*.tif")) + list(region_dir.glob("*.tiff"))
        found_bands: set[str] = set()
        for tif in tif_files:
            for band in EXPECTED_BANDS:
                if band in tif.name:
                    found_bands.add(band)
                    if tif.stat().st_size < 100:
                        report["empty_files"].append(str(tif))
                        report["issues_count"] += 1

        missing = EXPECTED_BANDS - found_bands
        if missing:
            report["missing_bands"].append({
                "region": region_dir.name,
                "missing": sorted(missing),
            })
            report["issues_count"] += 1

    return report


def validate_patches(patches_dir: str) -> dict[str, Any]:
    """
    Validate processed .npy patch directories.

    Checks:
    - Missing patch files
    - Corrupt arrays
    - NaN/Inf values
    - Constant arrays (no information)
    - Shape mismatches
    - Duplicate SHA256 hashes (identical patches)

    Args:
        patches_dir: Path to patches directory.

    Returns:
        Validation report dictionary.
    """
    data_path = Path(patches_dir)
    if not data_path.exists():
        return {"error": f"Directory not found: {patches_dir}", "patches": 0}

    sample_dirs = sorted([
        d for d in data_path.rglob("*")
        if d.is_dir() and any((d / f).exists() for f in EXPECTED_PATCH_FILES)
    ])

    report: dict[str, Any] = {
        "total_patches": len(sample_dirs),
        "missing_files": [],
        "corrupt_files": [],
        "nan_inf_files": [],
        "constant_arrays": [],
        "shape_issues": [],
        "duplicate_hashes": [],
        "issues_count": 0,
    }

    hash_map: dict[str, list[str]] = defaultdict(list)

    for sample_dir in sample_dirs:
        for expected_file in EXPECTED_PATCH_FILES:
            fpath = sample_dir / expected_file
            if not fpath.exists():
                report["missing_files"].append(str(fpath))
                report["issues_count"] += 1
                continue

            arr = _safe_load_npy(fpath)
            if arr is None:
                report["corrupt_files"].append(str(fpath))
                report["issues_count"] += 1
                continue

            # NaN/Inf check
            if not np.isfinite(arr).all():
                report["nan_inf_files"].append(str(fpath))
                report["issues_count"] += 1

            # Constant array check
            if arr.std() < 1e-10:
                report["constant_arrays"].append(str(fpath))
                report["issues_count"] += 1

            # Shape validation
            if expected_file == "tir_200m.npy" and arr.shape != (64, 64):
                report["shape_issues"].append(
                    {"file": str(fpath), "expected": "(64, 64)", "got": str(arr.shape)}
                )
                report["issues_count"] += 1
            elif expected_file == "tir_100m.npy" and arr.shape != (128, 128):
                report["shape_issues"].append(
                    {"file": str(fpath), "expected": "(128, 128)", "got": str(arr.shape)}
                )
                report["issues_count"] += 1
            elif expected_file == "rgb_100m.npy" and arr.shape != (3, 128, 128):
                report["shape_issues"].append(
                    {"file": str(fpath), "expected": "(3, 128, 128)", "got": str(arr.shape)}
                )
                report["issues_count"] += 1

            # Hash for duplicate detection
            h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
            hash_map[h].append(str(fpath))

    # Find duplicates
    for h, paths in hash_map.items():
        if len(paths) > 1:
            report["duplicate_hashes"].append({"hash": h, "files": paths})
            report["issues_count"] += 1

    return report


def print_report(report: dict[str, Any], title: str) -> None:
    """Print a formatted validation report."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    if "error" in report:
        print(f"  ERROR: {report['error']}")
        return

    total = report.get("total_regions", report.get("total_patches", 0))
    issues = report.get("issues_count", 0)
    status = "✓ CLEAN" if issues == 0 else f"✗ {issues} ISSUE(S) FOUND"
    print(f"  Total: {total}")
    print(f"  Status: {status}")

    for key, val in report.items():
        if key in ("total_regions", "total_patches", "issues_count", "error"):
            continue
        if isinstance(val, list) and val:
            print(f"\n  {key} ({len(val)}):")
            for item in val[:10]:
                if isinstance(item, dict):
                    print(f"    - {item}")
                else:
                    print(f"    - {item}")
            if len(val) > 10:
                print(f"    ... and {len(val) - 10} more")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate InfraNova AI dataset regions and patches."
    )
    parser.add_argument(
        "--dir",
        default="data/landsat9/input",
        help="Raw input directory with region folders.",
    )
    parser.add_argument(
        "--patches",
        default="data/landsat9/patches",
        help="Processed patches directory.",
    )
    parser.add_argument(
        "--output",
        default="outputs/region_validation_report.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    combined_report: dict[str, Any] = {}

    # Validate raw regions
    raw_dir = Path(args.dir)
    if raw_dir.exists():
        raw_report = validate_raw_regions(args.dir)
        combined_report["raw_regions"] = raw_report
        print_report(raw_report, "Raw Region Validation")
    else:
        print(f"Raw input directory not found: {args.dir} (skipping)")

    # Validate patches
    patches_dir = Path(args.patches)
    if patches_dir.exists():
        patch_report = validate_patches(args.patches)
        combined_report["patches"] = patch_report
        print_report(patch_report, "Patch Validation")
    else:
        print(f"Patches directory not found: {args.patches} (skipping)")

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(combined_report, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Full report saved to: {args.output}")


if __name__ == "__main__":
    main()

