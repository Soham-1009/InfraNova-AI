"""
Dataset Auditing Tool for InfraNova AI

Analyzes processed Landsat 9 patches to identify and classify blank, corrupted,
or low-variance samples. Produces a CSV report, a JSON summary, and visual grids
for anomalous categories.

Categories:
    VALID, ALL_ZERO, LOW_VARIANCE, HIGH_NODATA, CONTAINS_NAN, CONTAINS_INF, UNKNOWN
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "landsat9" / "patches"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Thresholds
LOW_VARIANCE_THRESHOLD = 1e-4
HIGH_NODATA_THRESHOLD = 0.50  # 50% of pixels are exactly zero

# Maximum samples to plot per anomaly category grid
MAX_GRID_SAMPLES = 16

# ---------------------------------------------------------------------------
# Enums & Types
# ---------------------------------------------------------------------------
class SampleClass(StrEnum):
    VALID = "VALID"
    ALL_ZERO = "ALL_ZERO"
    LOW_VARIANCE = "LOW_VARIANCE"
    HIGH_NODATA = "HIGH_NODATA"
    CONTAINS_NAN = "CONTAINS_NAN"
    CONTAINS_INF = "CONTAINS_INF"
    UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------
def load_sample(region_name: str, sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the numpy arrays for a given sample.

    Args:
        region_name: Name of the region.
        sample_id: Name of the sample directory (e.g., sample_000).

    Returns:
        Tuple of (tir_200m, tir_100m, rgb_100m).
    """
    sample_dir = DATA_DIR / region_name / sample_id

    tir_200m = np.load(sample_dir / "tir_200m.npy")
    tir_100m = np.load(sample_dir / "tir_100m.npy")
    rgb_100m = np.load(sample_dir / "rgb_100m.npy")

    return tir_200m, tir_100m, rgb_100m


def compute_statistics(tir_200m: np.ndarray, tir_100m: np.ndarray, rgb_100m: np.ndarray) -> dict[str, Any]:
    """
    Computes statistical metrics across all bands of the sample.
    Combines arrays into a single flattened array for global statistics.
    """
    # Flatten all arrays to compute global metrics for the sample
    flat_data = np.concatenate([
        tir_200m.flatten(),
        tir_100m.flatten(),
        rgb_100m.flatten()
    ])

    total_pixels = flat_data.size

    # Check for NaN and Inf before other metrics to avoid warnings
    nan_mask = np.isnan(flat_data)
    inf_mask = np.isinf(flat_data)

    nan_count = np.sum(nan_mask)
    inf_count = np.sum(inf_mask)

    # Mask out NaN/Inf for standard stats if present
    valid_data = flat_data[~(nan_mask | inf_mask)]

    if valid_data.size > 0:
        val_min = float(np.min(valid_data))
        val_max = float(np.max(valid_data))
        val_mean = float(np.mean(valid_data))
        val_std = float(np.std(valid_data))
        val_var = float(np.var(valid_data))

        # Zero threshold comparison (using strict equality for NoData detection)
        zero_count = np.sum(valid_data == 0.0)
        unique_vals = len(np.unique(valid_data))
    else:
        val_min = val_max = val_mean = val_std = val_var = 0.0
        zero_count = 0
        unique_vals = 0

    return {
        "Mean": val_mean,
        "Std": val_std,
        "Variance": val_var,
        "Zero %": (zero_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0,
        "NaN %": (nan_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0,
        "Inf %": (inf_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0,
        "Min": val_min,
        "Max": val_max,
        "Unique Values": unique_vals,
    }


def classify_sample(stats: dict[str, Any]) -> SampleClass:
    """
    Classifies a sample into an anomaly category based on its statistics.
    Order of checks determines precedence.
    """
    if stats["NaN %"] > 0:
        return SampleClass.CONTAINS_NAN

    if stats["Inf %"] > 0:
        return SampleClass.CONTAINS_INF

    if stats["Zero %"] == 100.0 or stats["Unique Values"] <= 1:
        return SampleClass.ALL_ZERO

    if stats["Zero %"] >= (HIGH_NODATA_THRESHOLD * 100.0):
        return SampleClass.HIGH_NODATA

    if stats["Variance"] < LOW_VARIANCE_THRESHOLD:
        return SampleClass.LOW_VARIANCE

    return SampleClass.VALID


def save_csv(results: list[dict[str, Any]], out_path: Path) -> None:
    """Saves the flat results list to a CSV report."""
    df = pd.DataFrame(results)
    # Reorder columns to ensure Region, Sample, and Classification are first
    cols = ["Region", "Sample", "Classification"]
    stats_cols = [c for c in df.columns if c not in cols]
    df = df[cols + stats_cols]

    df.to_csv(out_path, index=False)
    logger.info(f"CSV report saved to {out_path}")


def save_summary(results: list[dict[str, Any]], out_path: Path) -> None:
    """Generates and saves a JSON summary of the audit."""
    total = len(results)

    # Count classifications
    class_counts = Counter(r["Classification"] for r in results)

    # Calculate percentages
    percentages = {
        cls: round((count / total) * 100, 2) if total > 0 else 0.0
        for cls, count in class_counts.items()
    }

    summary = {
        "total_samples": total,
        "class_counts": dict(class_counts),
        "percentages": percentages,
        "thresholds": {
            "low_variance": LOW_VARIANCE_THRESHOLD,
            "high_nodata_percent": HIGH_NODATA_THRESHOLD * 100
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"JSON summary saved to {out_path}")


def generate_grids(results: list[dict[str, Any]], out_dir: Path) -> None:
    """
    Generates an image grid (up to MAX_GRID_SAMPLES) for each non-VALID category.
    """
    # Group by classification
    grouped = {cls.value: [] for cls in SampleClass if cls != SampleClass.VALID}

    for res in results:
        cls = res["Classification"]
        if cls != SampleClass.VALID.value:
            grouped[cls].append(res)

    for cls_name, samples in grouped.items():
        if not samples:
            continue

        # Take up to MAX_GRID_SAMPLES
        plot_samples = samples[:MAX_GRID_SAMPLES]
        n_samples = len(plot_samples)

        # Determine grid dimensions
        cols_per_sample = 2
        samples_per_row = 4

        rows = math.ceil(n_samples / samples_per_row)
        fig, axes = plt.subplots(rows, samples_per_row * cols_per_sample, figsize=(16, 4 * rows))

        # Ensure axes is 2D
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        fig.suptitle(f"Category: {cls_name} (Showing {n_samples} samples)", fontsize=16)

        # Hide all axes first
        for ax in axes.flat:
            ax.axis("off")

        for i, res in enumerate(plot_samples):
            r = i // samples_per_row
            c = (i % samples_per_row) * cols_per_sample

            try:
                _, tir_100m, rgb_100m = load_sample(res["Region"], res["Sample"])

                ax_rgb = axes[r, c]
                ax_tir = axes[r, c + 1]

                # Format RGB for display (CHW -> HWC)
                rgb_disp = np.transpose(rgb_100m, (1, 2, 0))
                # Normalize RGB to 0-1 for display if necessary
                rgb_disp = (rgb_disp - np.min(rgb_disp)) / (np.ptp(rgb_disp) + 1e-8)

                ax_rgb.imshow(rgb_disp)
                ax_rgb.set_title(f"{res['Region']}\n{res['Sample']} (RGB)", fontsize=8)
                ax_rgb.axis("off")

                # Format TIR for display
                ax_tir.imshow(tir_100m, cmap="inferno")
                ax_tir.set_title(f"{res['Region']}\n{res['Sample']} (TIR)", fontsize=8)
                ax_tir.axis("off")

            except Exception as e:
                logger.error(f"Error plotting {res['Region']}/{res['Sample']}: {e}")

        plt.tight_layout()
        out_file = out_dir / f"{cls_name}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated grid for {cls_name} -> {out_file.name}")


def main() -> None:
    logger.info("======================================")
    logger.info("DATASET AUDIT INITIATED")
    logger.info("======================================")

    if not DATA_DIR.exists():
        logger.error(f"Dataset directory not found: {DATA_DIR}")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Scan the dataset
    all_regions = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    sample_paths = []

    for region_dir in all_regions:
        for sample_dir in region_dir.iterdir():
            if sample_dir.is_dir() and sample_dir.name.startswith("sample_"):
                sample_paths.append((region_dir.name, sample_dir.name))

    total_samples = len(sample_paths)
    logger.info(f"Found {len(all_regions)} regions and {total_samples} samples.")

    results = []

    # 2. Process all samples
    logger.info("Analyzing samples...")
    for region_name, sample_id in tqdm(sample_paths, desc="Auditing"):
        try:
            tir_200m, tir_100m, rgb_100m = load_sample(region_name, sample_id)
            stats = compute_statistics(tir_200m, tir_100m, rgb_100m)
            classification = classify_sample(stats)

            # Build the result row
            row = {
                "Region": region_name,
                "Sample": sample_id,
                "Classification": classification.value,
                **stats
            }
            results.append(row)

        except Exception as e:
            logger.error(f"Failed to process {region_name}/{sample_id}: {e}")
            # Do not crash, classify as UNKNOWN
            row = {
                "Region": region_name,
                "Sample": sample_id,
                "Classification": SampleClass.UNKNOWN.value,
                "Mean": 0.0, "Std": 0.0, "Variance": 0.0,
                "Zero %": 0.0, "NaN %": 0.0, "Inf %": 0.0,
                "Min": 0.0, "Max": 0.0, "Unique Values": 0
            }
            results.append(row)

    # 3. Output artifacts
    logger.info("Generating reports and visualizations...")
    save_csv(results, REPORTS_DIR / "blank_patch_report.csv")
    save_summary(results, REPORTS_DIR / "blank_patch_summary.json")
    generate_grids(results, REPORTS_DIR)

    # 4. Final Console Report
    class_counts = Counter(r["Classification"] for r in results)

    print("\n======================================")
    print("DATASET AUDIT SUMMARY")
    print("======================================")
    print(f"{'Total samples':<20} {total_samples}")
    print("-" * 38)
    for cls in SampleClass:
        count = class_counts.get(cls.value, 0)
        print(f"{cls.value:<20} {count}")
    print("======================================\n")

    logger.info("Audit complete.")

if __name__ == "__main__":
    main()
