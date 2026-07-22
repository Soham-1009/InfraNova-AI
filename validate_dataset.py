"""
Validate Landsat 9 dataset patches for training readiness.

Reports missing files, corrupted data, wrong dimensions, NaN/Inf values,
duplicate samples, constant arrays, value range violations, and histogram stats.

Usage:
    python validate_dataset.py
    python validate_dataset.py --dir data/landsat9/splits/train
    python validate_dataset.py --dir data/landsat9/patches --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_FILES = ("tir_200m.npy", "tir_100m.npy", "rgb_100m.npy")
EXPECTED_SHAPES = {
    "tir_200m.npy": (64, 64),
    "tir_100m.npy": (128, 128),
    "rgb_100m.npy": (3, 128, 128),
}

# Reasonable value ranges for Landsat 9 data
VALUE_RANGES = {
    "tir_200m.npy": (-50.0, 100.0),     # Surface temperature in °C-ish
    "tir_100m.npy": (-50.0, 100.0),
    "rgb_100m.npy": (0.0, 1.0),          # Normalized reflectance
}


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()


def validate_sample(sample_dir: Path, verbose: bool = False) -> Tuple[List[str], Dict[str, str]]:
    """
    Validate a single sample directory.

    Returns:
        (issues, file_hashes) — issues is a list of error strings,
        file_hashes maps filename to SHA256.
    """
    issues: List[str] = []
    file_hashes: Dict[str, str] = {}

    for filename in EXPECTED_FILES:
        filepath = sample_dir / filename
        if not filepath.exists():
            issues.append(f"Missing {filename}")
            continue

        # Hash for duplicate detection
        file_hashes[filename] = compute_file_hash(filepath)

        try:
            arr = np.load(filepath)
        except Exception as exc:
            issues.append(f"Corrupted {filename}: {exc}")
            continue

        # Check shape
        expected = EXPECTED_SHAPES.get(filename)
        if expected is not None and arr.shape != expected:
            issues.append(f"{filename}: expected shape {expected}, got {arr.shape}")

        # Check empty
        if arr.size == 0:
            issues.append(f"{filename}: empty array (size=0)")
            continue

        # Check for NaN/Inf
        if not np.isfinite(arr).all():
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            issues.append(f"{filename}: {nan_count} NaN, {inf_count} Inf values")

        # Check constant array
        if np.std(arr) == 0:
            issues.append(f"{filename}: constant array (std=0, value={arr.flat[0]:.4f})")

        # Check all-zero (degenerate)
        if np.all(arr == 0):
            issues.append(f"{filename}: all-zero array")

        # Check value range
        vmin, vmax = VALUE_RANGES.get(filename, (None, None))
        if vmin is not None and vmax is not None:
            actual_min = float(np.min(arr))
            actual_max = float(np.max(arr))
            if actual_min < vmin * 10 or actual_max > vmax * 10:
                # Only warn for extreme outliers (10x beyond expected)
                issues.append(
                    f"{filename}: extreme values [{actual_min:.2f}, {actual_max:.2f}] "
                    f"(expected ~[{vmin}, {vmax}])"
                )

        # Verbose histogram stats
        if verbose and np.isfinite(arr).all():
            stats = (
                f"min={float(np.min(arr)):.4f}, max={float(np.max(arr)):.4f}, "
                f"mean={float(np.mean(arr)):.4f}, std={float(np.std(arr)):.4f}"
            )
            issues.append(f"[INFO] {filename}: {stats}")

    return issues, file_hashes


def validate_directory(
    data_dir: Path,
    verbose: bool = False,
) -> Tuple[int, int, Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Validate all samples in a dataset directory.

    Returns:
        (total_samples, valid_samples, errors_by_sample, hashes_by_sample)
    """
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return 0, 0, {}, {}

    # Discover sample directories
    sample_dirs: List[Path] = []

    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if any((child / f).exists() for f in EXPECTED_FILES):
            sample_dirs.append(child)
        else:
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and any((sub / f).exists() for f in EXPECTED_FILES):
                    sample_dirs.append(sub)

    if not sample_dirs:
        print(f"No sample directories found in {data_dir}")
        return 0, 0, {}, {}

    total = len(sample_dirs)
    errors_by_sample: Dict[str, List[str]] = {}
    hashes_by_sample: Dict[str, Dict[str, str]] = {}
    valid = 0

    for sample_dir in sample_dirs:
        issues, file_hashes = validate_sample(sample_dir, verbose=verbose)

        try:
            key = str(sample_dir.relative_to(data_dir))
        except ValueError:
            key = str(sample_dir)

        hashes_by_sample[key] = file_hashes

        # Filter out [INFO] lines for validity counting
        real_issues = [i for i in issues if not i.startswith("[INFO]")]
        if real_issues:
            errors_by_sample[key] = issues
        else:
            if verbose and issues:
                errors_by_sample[key] = issues  # Still store INFO lines
            valid += 1

    return total, valid, errors_by_sample, hashes_by_sample


def find_duplicate_hashes(
    hashes_by_sample: Dict[str, Dict[str, str]],
) -> List[str]:
    """Find samples with identical file content (SHA256 match)."""
    # Build composite hash per sample (hash of all file hashes)
    composite: Dict[str, List[str]] = {}
    for sample_key, file_hashes in hashes_by_sample.items():
        if len(file_hashes) == len(EXPECTED_FILES):
            combined = "|".join(file_hashes.get(f, "") for f in EXPECTED_FILES)
            composite.setdefault(combined, []).append(sample_key)

    duplicates: List[str] = []
    for combined_hash, samples in composite.items():
        if len(samples) > 1:
            duplicates.append(
                f"Identical content: {', '.join(samples[:5])}"
                f"{f' (+{len(samples)-5} more)' if len(samples) > 5 else ''}"
            )

    return duplicates


def check_data_leakage(data_dir: Path) -> List[str]:
    """Check for duplicate region names across splits."""
    region_to_splits: Dict[str, List[str]] = {}

    for split_name in ("train", "val", "test"):
        split_dir = data_dir / split_name
        if not split_dir.exists():
            continue
        for sample_dir in split_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            parts = sample_dir.name.split("_sample_")
            if parts:
                region = parts[0]
                region_to_splits.setdefault(region, []).append(split_name)

    duplicates: List[str] = []
    for region, splits in region_to_splits.items():
        unique_splits = set(splits)
        if len(unique_splits) > 1:
            duplicates.append(
                f"Region '{region}' appears in multiple splits: {sorted(unique_splits)}"
            )

    return duplicates


def print_summary_stats(hashes_by_sample: Dict[str, Dict[str, str]], data_dir: Path) -> None:
    """Print aggregate statistics across all samples."""
    print(f"\n{'=' * 60}")
    print("Aggregate Statistics")
    print(f"{'=' * 60}")

    for filename in EXPECTED_FILES:
        all_values = []
        for sample_key in hashes_by_sample:
            filepath = data_dir
            # Reconstruct path
            for part in sample_key.split("\\"):
                filepath = filepath / part
            npy_path = filepath / filename
            if npy_path.exists():
                try:
                    arr = np.load(npy_path)
                    if np.isfinite(arr).all():
                        all_values.append({
                            "min": float(np.min(arr)),
                            "max": float(np.max(arr)),
                            "mean": float(np.mean(arr)),
                            "std": float(np.std(arr)),
                        })
                except Exception:
                    pass

        if all_values:
            mins = [v["min"] for v in all_values]
            maxs = [v["max"] for v in all_values]
            means = [v["mean"] for v in all_values]
            stds = [v["std"] for v in all_values]
            print(f"\n  {filename} ({len(all_values)} samples):")
            print(f"    min:  {np.min(mins):.4f} .. {np.max(mins):.4f}")
            print(f"    max:  {np.min(maxs):.4f} .. {np.max(maxs):.4f}")
            print(f"    mean: {np.mean(means):.4f} ± {np.std(means):.4f}")
            print(f"    std:  {np.mean(stds):.4f} ± {np.std(stds):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Landsat 9 dataset patches for training readiness."
    )
    parser.add_argument(
        "--dir",
        default="data/landsat9/patches",
        help="Directory containing dataset samples to validate.",
    )
    parser.add_argument(
        "--check-splits",
        default="data/landsat9/splits",
        help="Directory containing train/val/test splits (for leakage check).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Report per-sample statistics and additional warnings.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print aggregate histogram statistics.",
    )
    args = parser.parse_args()

    data_dir = Path(args.dir)
    print(f"Validating dataset at: {data_dir}")
    print("=" * 60)

    total, valid, errors, hashes = validate_directory(data_dir, verbose=args.verbose)

    if total == 0:
        print("No samples found.")
        sys.exit(1)

    invalid = total - valid
    print(f"\nSamples scanned: {total}")
    print(f"  Valid:   {valid}")
    print(f"  Invalid: {invalid}")

    if errors:
        real_errors = {k: v for k, v in errors.items() if any(not i.startswith("[INFO]") for i in v)}
        if real_errors:
            print(f"\n--- Issues ({len(real_errors)} samples) ---")
            for sample, issues in sorted(real_errors.items()):
                print(f"\n  {sample}:")
                for issue in issues:
                    print(f"    - {issue}")

    # Duplicate detection
    duplicates = find_duplicate_hashes(hashes)
    if duplicates:
        print(f"\n--- Duplicate Samples ({len(duplicates)}) ---")
        for dup in duplicates:
            print(f"  - {dup}")

    # Data leakage check
    splits_dir = Path(args.check_splits)
    if splits_dir.exists():
        print(f"\n{'=' * 60}")
        print(f"Checking for data leakage in: {splits_dir}")
        leakage = check_data_leakage(splits_dir)
        if leakage:
            print(f"\n  WARNING: {len(leakage)} region(s) in multiple splits:")
            for leak in leakage:
                print(f"    - {leak}")
        else:
            print("  No data leakage detected.")

    # Aggregate stats
    if args.stats:
        print_summary_stats(hashes, data_dir)

    print(f"\n{'=' * 60}")
    if invalid == 0 and not duplicates:
        print("Dataset validation PASSED.")
    else:
        issues_found = []
        if invalid > 0:
            issues_found.append(f"{invalid} invalid sample(s)")
        if duplicates:
            issues_found.append(f"{len(duplicates)} duplicate group(s)")
        print(f"Dataset validation FAILED — {', '.join(issues_found)}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
