"""
Split region patch folders into train/val/test sets.

The split is region-level by default to avoid geography leakage: patches from
one source region must not appear in both train and validation/test sets.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List


PATCHES_DIR = Path("data/landsat9/patches")
OUTPUT_DIR = Path("data/landsat9/splits")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SPLITS = ("train", "val", "test")


def collect_region_dirs(patches_dir: Path) -> List[Path]:
    if not patches_dir.exists():
        raise FileNotFoundError(f"Patch directory not found: {patches_dir}")

    region_dirs = [
        region_dir
        for region_dir in sorted(patches_dir.iterdir())
        if region_dir.is_dir() and any(region_dir.glob("sample_*"))
    ]

    if not region_dirs:
        raise ValueError(f"No patch samples found in {patches_dir}")

    return region_dirs


def split_regions(region_dirs: List[Path], seed: int) -> Dict[str, List[Path]]:
    shuffled = region_dirs[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * TRAIN_RATIO)
    # FIX: Ensure at least 1 validation region to prevent crashes on tiny datasets
    n_val = max(1, int(n_total * VAL_RATIO)) 

    if n_total >= 3:
        train_dirs = shuffled[:n_train]
        val_dirs = shuffled[n_train : n_train + n_val]
        test_dirs = shuffled[n_train + n_val :]
        # Safety fallback if test_dirs becomes empty due to rounding
        if not test_dirs and len(val_dirs) > 1:
            test_dirs = [val_dirs.pop()]
    else:
        # Extreme edge case: less than 3 regions total
        train_dirs = shuffled
        val_dirs = shuffled[:1]  
        test_dirs = shuffled[-1:] 

    return {
        "train": train_dirs,
        "val": val_dirs,
        "test": test_dirs,
    }


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if overwrite:
            print(f"Overwriting existing splits in {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory {output_dir} already exists. Use --overwrite to replace it."
            )

    for split in SPLITS:
        (output_dir / split).mkdir(parents=True, exist_ok=True)


def copy_splits(region_splits: Dict[str, List[Path]], output_dir: Path) -> int:
    count = 0
    for split in SPLITS:
        split_dir = output_dir / split
        region_dirs = region_splits[split]

        for region_dir in region_dirs:
            for sample_dir in sorted(region_dir.glob("sample_*")):
                # Create a unique name to avoid collisions
                target_name = f"{region_dir.name}_{sample_dir.name}_{count:05d}"
                shutil.copytree(sample_dir, split_dir / target_name)
                count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Create region-level Landsat patch splits.")
    parser.add_argument("--patches-dir", default=str(PATCHES_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing split folders before creating new ones.",
    )
    args = parser.parse_args()

    patches_dir = Path(args.patches_dir)
    output_dir = Path(args.output_dir)

    region_dirs = collect_region_dirs(patches_dir)
    region_splits = split_regions(region_dirs, seed=args.seed)

    print(f"Regions found: {len(region_dirs)}")
    for split in SPLITS:
        sample_count = sum(1 for region_dir in region_splits[split] for _ in region_dir.glob("sample_*"))
        print(f"{split.title()}: {len(region_splits[split])} regions, {sample_count} patches")

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    total_copied = copy_splits(region_splits, output_dir)
    print(f"Successfully copied {total_copied} total patches.")


if __name__ == "__main__":
    main()