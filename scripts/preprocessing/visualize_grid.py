"""
Visualize Stratified Grid

Randomly samples 15 patches (stratified across different region prefixes/types if possible,
otherwise uniformly random) and generates a PNG grid of RGB and TIR images for visual inspection.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("c:/Users/soham/Desktop/Soham/InfraNova-AI")

def generate_visual_grid():
    patches_dir = PROJECT_ROOT / "data/landsat9/patches"
    output_png = PROJECT_ROOT / "data/landsat9/inspection_grid.png"

    if not patches_dir.exists():
        print(f"Patches directory not found: {patches_dir}")
        return

    all_samples = []

    for region_dir in sorted(patches_dir.iterdir()):
        if not region_dir.is_dir():
            continue
        for sample_dir in region_dir.iterdir():
            if sample_dir.is_dir():
                all_samples.append(sample_dir)

    if not all_samples:
        print("No patches found to visualize.")
        return

    # We will pick 15 random samples
    sample_size = min(15, len(all_samples))
    selected = random.sample(all_samples, sample_size)

    # 5 rows, 3 columns of pairs
    rows, cols = 5, 3
    fig, axes = plt.subplots(rows, cols * 2, figsize=(20, 15))
    fig.suptitle('Landsat 9 Patch Inspection (RGB | TIR)', fontsize=24)

    for idx, sample_path in enumerate(selected):
        if idx >= rows * cols:
            break

        r = idx // cols
        c = (idx % cols) * 2

        rgb = np.load(sample_path / "rgb_100m.npy")
        tir = np.load(sample_path / "tir_100m.npy")

        # Normalize RGB for display
        rgb_disp = np.transpose(rgb, (1, 2, 0)) # (H, W, 3)
        # 2nd to 98th percentile stretch for better visualization
        p2, p98 = np.percentile(rgb_disp, (2, 98))
        rgb_disp = np.clip((rgb_disp - p2) / (p98 - p2 + 1e-5), 0, 1)

        # Display RGB
        ax_rgb = axes[r, c]
        ax_rgb.imshow(rgb_disp)
        ax_rgb.set_title(f"{sample_path.parent.name}\n{sample_path.name} (RGB)", fontsize=10)
        ax_rgb.axis('off')

        # Display TIR
        ax_tir = axes[r, c+1]
        ax_tir.imshow(tir, cmap='inferno')
        ax_tir.set_title("TIR", fontsize=10)
        ax_tir.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Inspection grid saved to {output_png}")

if __name__ == "__main__":
    generate_visual_grid()
