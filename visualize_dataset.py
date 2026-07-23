"""
Visualize Landsat 9 dataset samples and model predictions.

Generates side-by-side grids: Thermal → RGB GT → Generated RGB → Difference.

Usage:
    python visualize_dataset.py
    python visualize_dataset.py --num-samples 8 --split test
    python visualize_dataset.py --no-model  # dataset-only visualization
    python visualize_dataset.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.landsat9_dataset import Landsat9Dataset


def load_generator(checkpoint_path: str, device: str = "cpu"):
    """Load the generator from a checkpoint."""
    from src.models.pix2pix.pix2pix import Pix2Pix
    from src.utils.checkpoint import load_torch_checkpoint

    ckpt = load_torch_checkpoint(checkpoint_path, map_location=device)
    model = Pix2Pix(device=device, in_channels=1, out_channels=3)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    return model


def to_display_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized RGB tensor from [-1, 1] to an HWC display array."""
    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError("Expected a batch containing exactly one RGB image")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3 or tensor.size(0) != 3:
        raise ValueError("Expected an RGB tensor shaped [3, H, W]")
    tensor = (tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
    return tensor.numpy().transpose(1, 2, 0)


@torch.inference_mode()
def visualize(
    split: str = "test",
    data_root: str = "data/landsat9/splits",
    checkpoint_path: str = "checkpoints/best/pix2pix_landsat_best.pth",
    num_samples: int = 6,
    output_dir: str = "outputs/visualizations",
    use_model: bool = True,
    image_size: int = 256,
) -> None:
    """Generate visualization grids."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    print(f"Loading {split} split from {data_root}")
    dataset = Landsat9Dataset(
        root_dir=data_root,
        split=split,
        image_size=image_size,
        augment=False,
    )

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    num_samples = min(num_samples, len(dataset))
    print(f"Visualizing {num_samples} samples")

    # Load model if requested
    model = None
    if use_model and Path(checkpoint_path).exists():
        print(f"Loading model: {checkpoint_path}")
        model = load_generator(checkpoint_path, device)
    elif use_model:
        print(f"Checkpoint not found: {checkpoint_path}. Skipping model predictions.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select evenly spaced samples
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    if model is not None:
        # Full visualization: Thermal → RGB GT → Generated → Difference
        ncols = 4
        col_labels = ["Thermal Input", "RGB Ground Truth", "Generated RGB", "Difference"]
    else:
        # Dataset-only: Thermal → RGB GT
        ncols = 2
        col_labels = ["Thermal Input", "RGB Ground Truth"]

    fig, axes = plt.subplots(
        num_samples, ncols,
        figsize=(ncols * 4, num_samples * 4),
        squeeze=False,
    )

    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        ir = sample["ir"]    # (1, H, W) or (C, H, W)
        rgb = sample["rgb"]  # (3, H, W)

        # Thermal image (squeeze to 2D for display)
        thermal_np = ((ir.squeeze(0).cpu().clamp(-1.0, 1.0) + 1.0) / 2.0).numpy()
        axes[row, 0].imshow(thermal_np, cmap="inferno")
        axes[row, 0].set_title(f"Sample {int(idx)}" if row == 0 else "")
        axes[row, 0].axis("off")

        # RGB ground truth
        rgb_np = to_display_rgb(rgb)
        axes[row, 1].imshow(rgb_np)
        axes[row, 1].axis("off")

        if model is not None:
            # Generate prediction
            ir_input = ir.unsqueeze(0).to(device)
            pred = model.generate(ir_input)
            pred_np = to_display_rgb(pred)

            axes[row, 2].imshow(pred_np)
            axes[row, 2].axis("off")

            # Difference map (absolute)
            diff = np.abs(pred_np - rgb_np)
            diff_mean = diff.mean(axis=-1)  # Average across channels
            im = axes[row, 3].imshow(diff_mean, cmap="hot", vmin=0, vmax=0.3)
            axes[row, 3].axis("off")

    # Column headers
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Save
    output_path = out_dir / f"visualization_{split}_{num_samples}samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Also save individual comparison strips
    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        ir = sample["ir"]
        rgb = sample["rgb"]

        fig_single, ax_row = plt.subplots(1, ncols, figsize=(ncols * 4, 4), squeeze=False)

        thermal_np = ((ir.squeeze(0).cpu().clamp(-1.0, 1.0) + 1.0) / 2.0).numpy()
        ax_row[0, 0].imshow(thermal_np, cmap="inferno")
        ax_row[0, 0].set_title("Thermal")
        ax_row[0, 0].axis("off")

        rgb_np = to_display_rgb(rgb)
        ax_row[0, 1].imshow(rgb_np)
        ax_row[0, 1].set_title("RGB GT")
        ax_row[0, 1].axis("off")

        if model is not None:
            ir_input = ir.unsqueeze(0).to(device)
            pred = model.generate(ir_input)
            pred_np = to_display_rgb(pred)

            ax_row[0, 2].imshow(pred_np)
            ax_row[0, 2].set_title("Generated")
            ax_row[0, 2].axis("off")

            diff = np.abs(pred_np - rgb_np).mean(axis=-1)
            ax_row[0, 3].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
            ax_row[0, 3].set_title("Difference")
            ax_row[0, 3].axis("off")

        plt.tight_layout()
        single_path = out_dir / f"sample_{int(idx):04d}.png"
        plt.savefig(single_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Individual samples saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize dataset samples and model predictions."
    )
    parser.add_argument(
        "--split", default="test",
        choices=["train", "val", "test"],
        help="Dataset split (default: test).",
    )
    parser.add_argument(
        "--data-root", default="data/landsat9/splits",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best/pix2pix_landsat_best.pth",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=6,
        help="Number of samples to visualize (default: 6).",
    )
    parser.add_argument(
        "--output-dir", default="outputs/visualizations",
        help="Output directory for visualizations.",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip model predictions (dataset-only visualization).",
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Image size (default: 256).",
    )
    args = parser.parse_args()

    visualize(
        split=args.split,
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        use_model=not args.no_model,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
