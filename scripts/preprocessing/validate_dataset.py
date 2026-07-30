"""
Dataset Integrity Validator

Verifies every generated patch pair (RGB / TIR) for:
- Dimension matching
- Correct dtypes
- Finite values (no NaNs or Infs)
- Variance (no blank/constant patches)
- Filename alignment
"""

import sys
from pathlib import Path
import numpy as np

# Make project root importable
PROJECT_ROOT = Path("c:/Users/soham/Desktop/Soham/InfraNova-AI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def validate_dataset():
    patches_dir = PROJECT_ROOT / "data/landsat9/patches"
    
    if not patches_dir.exists():
        print(f"Patches directory not found: {patches_dir}")
        return
        
    pairs_checked = 0
    corrupt = 0
    nan_count = 0
    blank = 0
    shape_mismatch = 0
    filename_mismatch = 0
    
    print(f"Validating patches in {patches_dir}...")
    
    for region_dir in sorted(patches_dir.iterdir()):
        if not region_dir.is_dir():
            continue
            
        for sample_dir in region_dir.iterdir():
            if not sample_dir.is_dir():
                continue
                
            rgb_path = sample_dir / "rgb_100m.npy"
            tir_path = sample_dir / "tir_100m.npy"
            tir_200_path = sample_dir / "tir_200m.npy"
            
            if not rgb_path.exists() or not tir_path.exists():
                filename_mismatch += 1
                continue
                
            try:
                rgb = np.load(rgb_path)
                tir = np.load(tir_path)
            except Exception as e:
                corrupt += 1
                continue
                
            pairs_checked += 1
            
            # Type and Finite Check
            if rgb.dtype != np.float32 or tir.dtype != np.float32:
                corrupt += 1
                
            if not np.isfinite(rgb).all() or not np.isfinite(tir).all():
                nan_count += 1
                continue
                
            # Shape Check (RGB is 3x128x128, TIR is 128x128)
            if rgb.shape != (3, 128, 128) or tir.shape != (128, 128):
                shape_mismatch += 1
                continue
                
            # Variance check (detect all-black or all-white patches)
            if np.var(rgb) < 1e-4 or np.var(tir) < 1e-4:
                blank += 1
                continue
                
    print("\nValidation Summary:")
    print("-" * 30)
    print(f"Pairs checked      : {pairs_checked:,}")
    print(f"Corrupt            : {corrupt:,}")
    print(f"NaN                : {nan_count:,}")
    print(f"Blank              : {blank:,}")
    print(f"Shape mismatch     : {shape_mismatch:,}")
    print(f"Filename mismatch  : {filename_mismatch:,}")
    print("-" * 30)
    
    if corrupt == 0 and nan_count == 0 and blank == 0 and shape_mismatch == 0 and filename_mismatch == 0:
        print("Dataset Integrity: PASSED")
    else:
        print("Dataset Integrity: FAILED")

if __name__ == "__main__":
    validate_dataset()
