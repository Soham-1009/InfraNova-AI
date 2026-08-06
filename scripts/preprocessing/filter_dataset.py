"""
Filter Dataset

Reads the blank patch report and removes samples classified as ALL_ZERO or HIGH_NODATA.
Moves them to a discarded folder to preserve the valid dataset before splitting.
"""

import shutil
from pathlib import Path

import pandas as pd


def filter_dataset():
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "reports" / "blank_patch_report.csv"
    patches_dir = root / "data" / "landsat9" / "patches"
    discard_dir = root / "data" / "landsat9" / "discarded"

    if not csv_path.exists():
        print("Report not found.")
        return

    df = pd.read_csv(csv_path)
    to_remove = df[df["Classification"].isin(["ALL_ZERO", "HIGH_NODATA"])]

    print(f"Found {len(to_remove)} patches to remove.")

    discard_dir.mkdir(parents=True, exist_ok=True)
    moved_count = 0

    for _, row in to_remove.iterrows():
        region = row["Region"]
        sample = row["Sample"]
        src = patches_dir / region / sample
        dst = discard_dir / region / sample

        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved_count += 1

    print(f"Successfully moved {moved_count} patches to {discard_dir}")

if __name__ == "__main__":
    filter_dataset()
