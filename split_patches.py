"""
Split patches into train/val/test sets.
"""

import shutil
from pathlib import Path
import random

PATCHES_DIR = Path("data/landsat9/patches")
OUTPUT_DIR = Path("data/landsat9/splits")

# 80/10/10 split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

random.seed(42)


def main():
    # Collect all patches with their source region
    all_patches = []
    
    for region_dir in PATCHES_DIR.iterdir():
        if not region_dir.is_dir():
            continue
        for sample_dir in region_dir.glob("sample_*"):
            all_patches.append((region_dir.name, sample_dir))
    
    print(f"Total patches found: {len(all_patches)}")
    
    # Shuffle
    random.shuffle(all_patches)
    
    # Split
    n_train = int(len(all_patches) * TRAIN_RATIO)
    n_val = int(len(all_patches) * VAL_RATIO)
    
    train_patches = all_patches[:n_train]
    val_patches = all_patches[n_train:n_train + n_val]
    test_patches = all_patches[n_train + n_val:]
    
    print(f"Train: {len(train_patches)}")
    print(f"Val:   {len(val_patches)}")
    print(f"Test:  {len(test_patches)}")
    
    # Create split folders with symlinks (or copies)
    for split_name, patches in [
        ('train', train_patches),
        ('val', val_patches),
        ('test', test_patches),
    ]:
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (region, sample_dir) in enumerate(patches):
            target_name = f"{region}_{sample_dir.name}_{idx:05d}"
            target_dir = split_dir / target_name
            
            if target_dir.exists():
                continue
            
            # Copy patch folder
            shutil.copytree(sample_dir, target_dir)
        
        print(f"  {split_name}: {len(list(split_dir.iterdir()))} folders")
    
    print("\nDone")


if __name__ == '__main__':
    main()