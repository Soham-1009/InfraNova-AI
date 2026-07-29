"""
Compare two InfraNova AI model checkpoints side by side.

Runs inference with both checkpoints on the same input(s) and produces
side-by-side visual comparisons and metric summaries.

Usage:
    python compare_models.py --ckpt-a checkpoints/best/old.pth --ckpt-b checkpoints/best/new.pth
    python compare_models.py --ckpt-a ckpt_a.pth --ckpt-b ckpt_b.pth --input data/landsat9/splits
    python compare_models.py --help
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
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


def _to_01(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(-1, 1) + 1.0) / 2.0


def _psnr_np(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_np(pred: np.ndarray, target: np.ndarray) -> float:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_p = np.mean(pred)
    mu_t = np.mean(target)
    sp = np.std(pred)
    st = np.std(target)
    sc = np.mean((pred - mu_p) * (target - mu_t))
    num = (2 * mu_p * mu_t + C1) * (2 * sc + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sp ** 2 + st ** 2 + C2)
    return float(num / den)


def load_model(ckpt_path: str, device: str) -> Pix2Pix:
    """Load a Pix2Pix model from a checkpoint."""
    ckpt = load_torch_checkpoint(ckpt_path, map_location=device)
    model = Pix2Pix(device=device, in_channels=1, out_channels=3)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


@torch.inference_mode()
def compare_models(
    ckpt_a_path: str,
    ckpt_b_path: str,
    data_root: str = str(PROJECT_ROOT / "data/landsat9/splits"),
    split: str = "val",
    image_size: int = 256,
    num_samples: int = 8,
    output_dir: str = "outputs/model_comparison",
    name_a: str = "Model A",
    name_b: str = "Model B",
) -> None:
    """Run inference with both models and compare."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print(f"Loading {name_a}: {ckpt_a_path}")
    model_a = load_model(ckpt_a_path, device)
    print(f"Loading {name_b}: {ckpt_b_path}")
    model_b = load_model(ckpt_b_path, device)

    # Load dataset
    dataset = Landsat9Dataset(
        root_dir=data_root,
        split=split,
        image_size=image_size,
        augment=False,
    )
    print(f"Samples: {len(dataset)}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n = min(num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    results: list[dict[str, float]] = []

    for panel_idx, idx in enumerate(indices):
        sample = dataset[idx]
        ir = sample["ir"].unsqueeze(0).to(device)
        rgb_gt = sample["rgb"].unsqueeze(0).to(device)

        # Time inference
        t0 = time.perf_counter()
        pred_a = model_a.generate(ir)
        time_a = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_b = model_b.generate(ir)
        time_b = time.perf_counter() - t0

        # Convert to numpy [0,1]
        ir_np = _to_01(ir).squeeze(0).cpu().numpy()[0]
        gt_np = _to_01(rgb_gt).squeeze(0).cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        pred_a_np = _to_01(pred_a).squeeze(0).cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        pred_b_np = _to_01(pred_b).squeeze(0).cpu().numpy().transpose(1, 2, 0).clip(0, 1)

        # Metrics
        row = {
            "sample_idx": int(idx),
            f"{name_a}_psnr": _psnr_np(pred_a_np, gt_np),
            f"{name_a}_ssim": _ssim_np(pred_a_np, gt_np),
            f"{name_a}_time_ms": time_a * 1000,
            f"{name_b}_psnr": _psnr_np(pred_b_np, gt_np),
            f"{name_b}_ssim": _ssim_np(pred_b_np, gt_np),
            f"{name_b}_time_ms": time_b * 1000,
        }
        results.append(row)

        # Visual comparison
        _fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(ir_np, cmap="inferno")
        axes[0].set_title("Thermal Input", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_np)
        axes[1].set_title("Ground Truth", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_a_np)
        axes[2].set_title(
            f"{name_a}\nSSIM={row[f'{name_a}_ssim']:.4f} PSNR={row[f'{name_a}_psnr']:.1f}",
            fontsize=10,
        )
        axes[2].axis("off")

        axes[3].imshow(pred_b_np)
        axes[3].set_title(
            f"{name_b}\nSSIM={row[f'{name_b}_ssim']:.4f} PSNR={row[f'{name_b}_psnr']:.1f}",
            fontsize=10,
        )
        axes[3].axis("off")

        plt.tight_layout()
        save_path = out_path / f"comparison_{panel_idx:03d}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Model Comparison Summary ({n} samples)")
    print(f"{'=' * 60}")

    for metric in ["psnr", "ssim", "time_ms"]:
        vals_a = [r[f"{name_a}_{metric}"] for r in results]
        vals_b = [r[f"{name_b}_{metric}"] for r in results]
        print(f"  {metric.upper():>8s} | {name_a}: {np.mean(vals_a):.4f} ± {np.std(vals_a):.4f} | "
              f"{name_b}: {np.mean(vals_b):.4f} ± {np.std(vals_b):.4f}")

    # Save CSV
    csv_path = out_path / "comparison_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two InfraNova AI model checkpoints."
    )
    parser.add_argument("--ckpt-a", required=True, help="Path to first checkpoint.")
    parser.add_argument("--ckpt-b", required=True, help="Path to second checkpoint.")
    parser.add_argument("--name-a", default="Model A", help="Name for first model.")
    parser.add_argument("--name-b", default="Model B", help="Name for second model.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data/landsat9/splits"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", default="outputs/model_comparison")
    args = parser.parse_args()

    for ckpt_path, name in [(args.ckpt_a, args.name_a), (args.ckpt_b, args.name_b)]:
        if not Path(ckpt_path).exists():
            print(f"Checkpoint not found for {name}: {ckpt_path}")
            sys.exit(1)

    compare_models(
        ckpt_a_path=args.ckpt_a,
        ckpt_b_path=args.ckpt_b,
        data_root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        num_samples=args.samples,
        output_dir=args.output,
        name_a=args.name_a,
        name_b=args.name_b,
    )


if __name__ == "__main__":
    main()


