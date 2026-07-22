"""
Generate aggregate dataset statistics for InfraNova AI.

Scans dataset patches and produces a dataset_statistics.json with
counts, per-band statistics, missing values, and corrupted files.

Usage:
    python dataset_statistics.py
    python dataset_statistics.py --dir data/landsat9/splits
    python dataset_statistics.py --output dataset_statistics.json
    python dataset_statistics.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

EXPECTED_FILES = ("tir_200m.npy", "tir_100m.npy", "rgb_100m.npy")


def discover_samples(data_dir: Path) -> Dict[str, List[Path]]:
    """Discover sample directories grouped by split."""
    groups: Dict[str, List[Path]] = {}

    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        # Check if this is a split directory (train/val/test)
        if child.name in ("train", "val", "test"):
            samples = sorted(
                d for d in child.iterdir()
                if d.is_dir() and any((d / f).exists() for f in EXPECTED_FILES)
            )
            if samples:
                groups[child.name] = samples
        else:
            # Direct sample directory
            if any((child / f).exists() for f in EXPECTED_FILES):
                groups.setdefault("all", []).append(child)
            else:
                # Nested region/sample structure
                for sub in sorted(child.iterdir()):
                    if sub.is_dir() and any((sub / f).exists() for f in EXPECTED_FILES):
                        groups.setdefault("all", []).append(sub)

    return groups


def compute_band_stats(arrays: List[np.ndarray]) -> Dict[str, Any]:
    """Compute aggregate statistics across a list of arrays."""
    if not arrays:
        return {}

    all_finite = [a[np.isfinite(a)] for a in arrays]
    combined = np.concatenate(all_finite) if all_finite else np.array([])

    if combined.size == 0:
        return {"count": len(arrays), "error": "no finite values"}

    return {
        "count": len(arrays),
        "shape": list(arrays[0].shape),
        "dtype": str(arrays[0].dtype),
        "min": float(np.min(combined)),
        "max": float(np.max(combined)),
        "mean": float(np.mean(combined)),
        "std": float(np.std(combined)),
        "median": float(np.median(combined)),
        "p2": float(np.percentile(combined, 2)),
        "p98": float(np.percentile(combined, 98)),
        "nan_count": sum(int(np.isnan(a).sum()) for a in arrays),
        "inf_count": sum(int(np.isinf(a).sum()) for a in arrays),
        "zero_arrays": sum(1 for a in arrays if np.all(a == 0)),
        "constant_arrays": sum(1 for a in arrays if np.std(a) == 0),
    }


def generate_statistics(
    data_dir: str = "data/landsat9/splits",
    output_path: str = "dataset_statistics.json",
) -> Dict[str, Any]:
    """Generate comprehensive dataset statistics."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory not found: {data_path}")
        sys.exit(1)

    groups = discover_samples(data_path)
    if not groups:
        print(f"No sample directories found in {data_path}")
        sys.exit(1)

    stats: Dict[str, Any] = {
        "data_dir": str(data_path),
        "splits": {},
    }

    # Extract region names
    all_regions = set()
    total_patches = 0

    for split_name, samples in groups.items():
        split_stats: Dict[str, Any] = {"num_samples": len(samples)}
        total_patches += len(samples)

        # Collect arrays per file type
        arrays_by_file: Dict[str, List[np.ndarray]] = {f: [] for f in EXPECTED_FILES}
        corrupted = 0
        missing = Counter()

        for sample_dir in samples:
            # Extract region name
            parts = sample_dir.name.split("_sample_")
            if parts:
                all_regions.add(parts[0])

            for filename in EXPECTED_FILES:
                filepath = sample_dir / filename
                if not filepath.exists():
                    missing[filename] += 1
                    continue
                try:
                    arr = np.load(filepath)
                    arrays_by_file[filename].append(arr)
                except Exception:
                    corrupted += 1

        split_stats["corrupted_files"] = corrupted
        split_stats["missing_files"] = dict(missing)

        # Per-file statistics
        for filename in EXPECTED_FILES:
            if arrays_by_file[filename]:
                split_stats[filename] = compute_band_stats(arrays_by_file[filename])

        stats["splits"][split_name] = split_stats

    stats["total_regions"] = len(all_regions)
    stats["total_patches"] = total_patches
    stats["regions"] = sorted(all_regions)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    # Print summary
    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Data dir:      {data_path}")
    print(f"Total regions: {stats['total_regions']}")
    print(f"Total patches: {stats['total_patches']}")
    print()

    for split_name, split_stats in stats["splits"].items():
        print(f"  {split_name}: {split_stats['num_samples']} samples")
        if split_stats.get("corrupted_files"):
            print(f"    Corrupted: {split_stats['corrupted_files']}")
        if split_stats.get("missing_files"):
            for fname, count in split_stats["missing_files"].items():
                print(f"    Missing {fname}: {count}")
        for filename in EXPECTED_FILES:
            if filename in split_stats:
                fs = split_stats[filename]
                print(
                    f"    {filename}: "
                    f"mean={fs.get('mean', 0):.4f} ± {fs.get('std', 0):.4f}, "
                    f"range=[{fs.get('min', 0):.4f}, {fs.get('max', 0):.4f}]"
                )

    print(f"\nSaved to: {output_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate aggregate dataset statistics."
    )
    parser.add_argument(
        "--dir",
        default="data/landsat9/splits",
        help="Dataset directory to scan.",
    )
    parser.add_argument(
        "--output",
        default="dataset_statistics.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    generate_statistics(data_dir=args.dir, output_path=args.output)


if __name__ == "__main__":
    main()
