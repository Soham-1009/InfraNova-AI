"""
Final cleanup of unused files and folders.
"""

import os
import shutil
from pathlib import Path

files_to_delete = [
    # Old checkpoint
    'checkpoints/best/pix2pix_best.pth',
    
    # Old visualizations
    'outputs/visualizations/dataset_test.png',
    
    # Old training artifacts
    'logs/loss_curve.png',
    'logs/training.csv',
    
    # Typo file
    'docs/demo_stratergy.md',
    
    # Unused detection
    'src/detection/detector.py',
    'src/detection/yolo_compare.py',
    
    # Unused evaluation
    'src/evaluation/fid_metric.py',
    'src/evaluation/lpips_metric.py',
    'src/evaluation/metrics.py',
    'src/evaluation/psnr.py',
    'src/evaluation/ssim.py',
    
    # Old inference
    'src/inference/colorize.py',
    'src/inference/enhance.py',
    'src/inference/predictor.py',
    
    # Old training
    'src/training/train.py',
    
    # Old tests
    'tests/test_dataset.py',
    'tests/test_inference.py',
    'tests/test_metrics.py',
    'tests/test_model.py',
    
    # Unused utilities
    'src/utils/config_loader.py',
    'src/utils/seed.py',
    'src/utils/visualization.py',
    
    # Old transforms
    'src/datasets/transforms.py',
    
    # Old configs
    'configs/model.yaml',
    'configs/paths.yaml',
]

# Old visualizations (pattern match)
viz_dir = Path('outputs/visualizations')
if viz_dir.exists():
    for f in viz_dir.glob('epoch_*.png'):
        files_to_delete.append(str(f))

folders_to_delete = [
    'data/processed',
    'data/raw',
    'data/metadata',
    'docs/figures',
    'logs/tensorboard',
    'logs/wandb',
    'outputs/predictions',
    'outputs/reports',
    'src/models/common',
    'src/models/cyclegan',
    'tests',
]

print("Cleanup starting...")
print("=" * 50)

deleted_files = 0
deleted_folders = 0

for f in files_to_delete:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"  Deleted: {f}")
            deleted_files += 1
        except Exception as e:
            print(f"  Error: {f} - {e}")

for folder in folders_to_delete:
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            print(f"  Deleted folder: {folder}")
            deleted_folders += 1
        except Exception as e:
            print(f"  Error: {folder} - {e}")

print("=" * 50)
print(f"Deleted {deleted_files} files, {deleted_folders} folders")