from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "blanchon/EuroSAT_MSI"
OUTPUT_ROOT = Path("data/multispectral")

EXPECTED_BANDS = 13
EXPECTED_HW = (64, 64)


def extract_array(sample: Any) -> np.ndarray:
    """Convert sample to (13, 64, 64) uint16 array."""
    if isinstance(sample, dict):
        if "image" in sample:
            arr = np.array(sample["image"])
        else:
            raise ValueError("No 'image' key in sample")
    else:
        arr = np.array(sample)
    
    arr = np.asarray(arr)
    
    if arr.ndim == 3 and arr.shape[-1] == EXPECTED_BANDS:
        arr = np.transpose(arr, (2, 0, 1))
    
    if arr.shape[0] != EXPECTED_BANDS or arr.shape[1:] != EXPECTED_HW:
        raise ValueError(f"Invalid shape: {arr.shape}")
    
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    
    return arr


def save_split(samples, split_name: str, max_samples: int) -> int:
    """Save split to disk."""
    split_dir = OUTPUT_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    n = min(max_samples, len(samples))
    
    for idx in tqdm(range(n), desc=f"Saving {split_name}"):
        try:
            arr = extract_array(samples[idx])
            out_path = split_dir / f"{idx:06d}.npy"
            np.save(out_path, arr)
            saved += 1
        except Exception as exc:
            print(f"Skipping {idx}: {exc}")
    
    return saved


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("Loading EuroSAT MSI...")
    ds_train = load_dataset(DATASET_NAME, split="train")
    ds_val = load_dataset(DATASET_NAME, split="validation")
    ds_test = load_dataset(DATASET_NAME, split="test")
    
    print(f"\nDataset sizes: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")
    
    print("\nSaving train split...")
    n_train = save_split(ds_train, "train", 16200)
    
    print("\nSaving val split...")
    n_val = save_split(ds_val, "val", 5400)
    
    print("\nSaving test split...")
    n_test = save_split(ds_test, "test", 5400)
    
    print(f"\nDone. Train: {n_train}, Val: {n_val}, Test: {n_test}")


if __name__ == "__main__":
    main()