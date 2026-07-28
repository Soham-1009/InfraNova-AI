"""
Visualize side-by-side comparison of InfraNova AI predictions.

Generates publication-quality panels showing:
- Thermal input (colormapped)
- Ground truth RGB
- Generated RGB
- Absolute difference map
- Saturation map (per-pixel channel std)

Usage:
    python visualize_comparison.py --checkpoint checkpoints/best/pix2pix_landsat_best.pth
    python visualize_comparison.py --samples 8
    python visualize_comparison.py --split val --output outputs/comparison
    python visualize_comparison.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.landsat9_dataset import Landsat9Dataset
from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_torch_checkpoint


def _to_01(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize [-1,1] to [0,1]."""
    return (tensor.clamp(-1, 1) + 1.0) / 2.0


@torch.inference_mode()
def generate_comparison(
    checkpoint_path: str,
    split: str = "val",
    data_root: str = str(PROJECT_ROOT / "data/landsat9/splits"),
    image_size: int = 256,
    num_samples: int = 4,
    output_dir: str = "outputs/comparison",
) -> None:
    """Generate side-by-side comparison panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = load_torch_checkpoint(checkpoint_path, map_location=device)
    model = Pix2Pix(device=device, in_channels=1, out_channels=3)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Load dataset
    dataset = Landsat9Dataset(
        root_dir=data_root,
        split=split,
        image_size=image_size,
        augment=False,
    )
    print(f"Samples available: {len(dataset)}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n = min(num_samples, len(dataset))
    # Spread samples evenly across the dataset
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    for panel_idx, idx in enumerate(indices):
        sample = dataset[idx]
        ir = sample["ir"].unsqueeze(0).to(device)
        rgb_gt = sample["rgb"].unsqueeze(0).to(device)

        rgb_pred = model.generate(ir)

        ir_01 = _to_01(ir).squeeze(0).cpu().numpy()
        gt_01 = _to_01(rgb_gt).squeeze(0).cpu().numpy()
        pred_01 = _to_01(rgb_pred).squeeze(0).cpu().numpy()

        # Convert CHW to HWC for display
        ir_disp = ir_01[0]  # single channel
        gt_disp = np.transpose(gt_01, (1, 2, 0)).clip(0, 1)
        pred_disp = np.transpose(pred_01, (1, 2, 0)).clip(0, 1)

        # Difference map (mean absolute error per pixel, normalized)
        diff = np.abs(pred_disp - gt_disp).mean(axis=-1)

        # Saturation map (channel std per pixel)
        pred_sat = np.std(pred_disp, axis=-1)
        gt_sat = np.std(gt_disp, axis=-1)

        # Create panel
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        axes[0].imshow(ir_disp, cmap="inferno")
        axes[0].set_title("Thermal Input", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_disp)
        axes[1].set_title("Ground Truth RGB", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_disp)
        axes[2].set_title("Generated RGB", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        im_diff = axes[3].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
        axes[3].set_title("Error Map", fontsize=12, fontweight="bold")
        axes[3].axis("off")
        plt.colorbar(im_diff, ax=axes[3], fraction=0.046, pad=0.04)

        # Saturation comparison: show predicted saturation
        im_sat = axes[4].imshow(pred_sat, cmap="viridis", vmin=0, vmax=0.25)
        axes[4].set_title("Pred Saturation", fontsize=12, fontweight="bold")
        axes[4].axis("off")
        plt.colorbar(im_sat, ax=axes[4], fraction=0.046, pad=0.04)

        plt.suptitle(f"Sample {idx} — {sample.get('name', idx)}", fontsize=14, y=1.02)
        plt.tight_layout()

        save_path = out_path / f"comparison_{panel_idx:03d}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # Also save a combined grid with all samples
    if n > 1:
        fig, axes = plt.subplots(n, 5, figsize=(25, 5 * n), squeeze=False)

        for row, idx in enumerate(indices):
            sample = dataset[idx]
            ir = sample["ir"].unsqueeze(0).to(device)
            rgb_gt = sample["rgb"].unsqueeze(0).to(device)
            rgb_pred = model.generate(ir)

            ir_01 = _to_01(ir).squeeze(0).cpu().numpy()
            gt_01 = _to_01(rgb_gt).squeeze(0).cpu().numpy()
            pred_01 = _to_01(rgb_pred).squeeze(0).cpu().numpy()

            ir_disp = ir_01[0]
            gt_disp = np.transpose(gt_01, (1, 2, 0)).clip(0, 1)
            pred_disp = np.transpose(pred_01, (1, 2, 0)).clip(0, 1)
            diff = np.abs(pred_disp - gt_disp).mean(axis=-1)
            pred_sat = np.std(pred_disp, axis=-1)

            axes[row, 0].imshow(ir_disp, cmap="inferno")
            axes[row, 0].axis("off")
            axes[row, 1].imshow(gt_disp)
            axes[row, 1].axis("off")
            axes[row, 2].imshow(pred_disp)
            axes[row, 2].axis("off")
            axes[row, 3].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
            axes[row, 3].axis("off")
            axes[row, 4].imshow(pred_sat, cmap="viridis", vmin=0, vmax=0.25)
            axes[row, 4].axis("off")

            if row == 0:
                for col, title in enumerate(["Thermal", "Ground Truth", "Generated", "Error", "Saturation"]):
                    axes[0, col].set_title(title, fontsize=11, fontweight="bold")

        plt.tight_layout()
        grid_path = out_path / "comparison_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Combined grid saved: {grid_path}")

    print(f"\nDone. {n} comparison panels saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual comparison panels for InfraNova AI."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best/pix2pix_landsat_best.pth")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data/landsat9/splits"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", default="outputs/comparison")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    generate_comparison(
        checkpoint_path=args.checkpoint,
        split=args.split,
        data_root=args.data_root,
        image_size=args.image_size,
        num_samples=args.samples,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()


