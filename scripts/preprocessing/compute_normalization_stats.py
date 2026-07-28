
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""
Compute global normalization statistics across all dataset patches.

Scans all .npy patches and computes per-band statistics (mean, std,
percentiles) that should be used for consistent normalization instead
of per-sample percentile stretching.

Usage:
    python compute_normalization_stats.py
    python compute_normalization_stats.py --dir data/landsat9/patches
    python compute_normalization_stats.py --output configs/normalization_stats.json
    python compute_normalization_stats.py --help
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np

EXPECTED_FILES = ("tir_100m.npy", "rgb_100m.npy")


def collect_stats(data_dir: str = "data/landsat9/patches") -> Dict[str, Any]:
    """
    Scan all sample directories and compute global per-band statistics.

    Returns a dict suitable for JSON serialization with keys:
        tir_100m: {mean, std, p2, p98, min, max, count}
        rgb_100m: {per_band: [{mean, std, p2, p98, min, max}] for R,G,B}
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory not found: {data_path}")
        sys.exit(1)

    # Discover sample directories (supports both flat and split layouts)
    sample_dirs: List[Path] = []

    for child in sorted(data_path.rglob("*")):
        if child.is_dir() and any((child / f).exists() for f in EXPECTED_FILES):
            sample_dirs.append(child)

    if not sample_dirs:
        print(f"No sample directories found in {data_path}")
        sys.exit(1)

    print(f"Found {len(sample_dirs)} sample directories")

    # Accumulate values using Welford's online algorithm for memory efficiency
    # For percentiles, we need to store per-file stats and combine

    tir_values: List[float] = []
    tir_means: List[float] = []
    tir_stds: List[float] = []

    rgb_means: List[List[float]] = [[], [], []]  # R, G, B
    rgb_stds: List[List[float]] = [[], [], []]
    rgb_mins: List[List[float]] = [[], [], []]
    rgb_maxs: List[List[float]] = [[], [], []]

    # For percentiles: subsample to keep memory reasonable
    tir_subsample: List[float] = []
    rgb_subsamples: List[List[float]] = [[], [], []]
    subsample_rate = max(1, len(sample_dirs) // 500)  # Keep ~500 samples for percentiles

    for idx, sample_dir in enumerate(sample_dirs):
        # TIR
        tir_path = sample_dir / "tir_100m.npy"
        if tir_path.exists():
            try:
                arr = np.load(tir_path).astype(np.float64)
                if np.isfinite(arr).all() and arr.size > 0:
                    tir_means.append(float(np.mean(arr)))
                    tir_stds.append(float(np.std(arr)))
                    if idx % subsample_rate == 0:
                        # Subsample pixels for percentile computation
                        flat = arr.flatten()
                        if len(flat) > 1000:
                            indices = np.random.choice(len(flat), 1000, replace=False)
                            tir_subsample.extend(flat[indices].tolist())
                        else:
                            tir_subsample.extend(flat.tolist())
            except Exception:
                pass

        # RGB
        rgb_path = sample_dir / "rgb_100m.npy"
        if rgb_path.exists():
            try:
                arr = np.load(rgb_path).astype(np.float64)
                if np.isfinite(arr).all() and arr.size > 0 and arr.shape[0] == 3:
                    for band_idx in range(3):
                        band = arr[band_idx]
                        rgb_means[band_idx].append(float(np.mean(band)))
                        rgb_stds[band_idx].append(float(np.std(band)))
                        rgb_mins[band_idx].append(float(np.min(band)))
                        rgb_maxs[band_idx].append(float(np.max(band)))
                        if idx % subsample_rate == 0:
                            flat = band.flatten()
                            if len(flat) > 1000:
                                indices = np.random.choice(len(flat), 1000, replace=False)
                                rgb_subsamples[band_idx].extend(flat[indices].tolist())
                            else:
                                rgb_subsamples[band_idx].extend(flat.tolist())
            except Exception:
                pass

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(sample_dirs)} samples...")

    # Compute global statistics
    stats: Dict[str, Any] = {"sample_count": len(sample_dirs)}

    # TIR stats
    if tir_means:
        tir_sub = np.array(tir_subsample)
        stats["tir_100m"] = {
            "global_mean": float(np.mean(tir_means)),
            "global_std": float(np.sqrt(np.mean(np.array(tir_stds) ** 2 + np.array(tir_means) ** 2) - np.mean(tir_means) ** 2)),
            "p2": float(np.percentile(tir_sub, 2)) if len(tir_sub) > 0 else 0.0,
            "p98": float(np.percentile(tir_sub, 98)) if len(tir_sub) > 0 else 1.0,
            "p1": float(np.percentile(tir_sub, 1)) if len(tir_sub) > 0 else 0.0,
            "p99": float(np.percentile(tir_sub, 99)) if len(tir_sub) > 0 else 1.0,
            "count": len(tir_means),
        }

    # RGB stats (per band: R=0, G=1, B=2)
    band_names = ["red", "green", "blue"]
    if rgb_means[0]:
        rgb_stats = {}
        for band_idx, name in enumerate(band_names):
            sub = np.array(rgb_subsamples[band_idx])
            means_arr = np.array(rgb_means[band_idx])
            stds_arr = np.array(rgb_stds[band_idx])
            # Combined std using parallel algorithm
            combined_std = float(np.sqrt(
                np.mean(stds_arr ** 2 + means_arr ** 2) - np.mean(means_arr) ** 2
            ))
            rgb_stats[name] = {
                "global_mean": float(np.mean(means_arr)),
                "global_std": combined_std,
                "p2": float(np.percentile(sub, 2)) if len(sub) > 0 else 0.0,
                "p98": float(np.percentile(sub, 98)) if len(sub) > 0 else 1.0,
                "p1": float(np.percentile(sub, 1)) if len(sub) > 0 else 0.0,
                "p99": float(np.percentile(sub, 99)) if len(sub) > 0 else 1.0,
            }
        rgb_stats["count"] = len(rgb_means[0])
        stats["rgb_100m"] = rgb_stats

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute global normalization statistics for the dataset."
    )
    parser.add_argument(
        "--dir",
        default="data/landsat9/patches",
        help="Root directory containing sample patches.",
    )
    parser.add_argument(
        "--output",
        default="configs/normalization_stats.json",
        help="Output JSON file for statistics.",
    )
    args = parser.parse_args()

    print(f"Scanning: {args.dir}")
    stats = collect_stats(args.dir)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("Global Normalization Statistics")
    print(f"{'=' * 60}")
    print(f"Samples: {stats['sample_count']}")

    if "tir_100m" in stats:
        t = stats["tir_100m"]
        print(f"\nTIR 100m ({t['count']} samples):")
        print(f"  mean={t['global_mean']:.4f}, std={t['global_std']:.4f}")
        print(f"  p2={t['p2']:.4f}, p98={t['p98']:.4f}")

    if "rgb_100m" in stats:
        r = stats["rgb_100m"]
        print(f"\nRGB 100m ({r['count']} samples):")
        for name in ["red", "green", "blue"]:
            b = r[name]
            print(f"  {name}: mean={b['global_mean']:.4f}, std={b['global_std']:.4f}, p2={b['p2']:.4f}, p98={b['p98']:.4f}")

    print(f"\nSaved to: {args.output}")
    print("\nNext: set 'normalization.stats_file' in config.yaml to use these stats during training.")


if __name__ == "__main__":
    main()

