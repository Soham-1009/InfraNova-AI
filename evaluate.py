"""
Evaluate InfraNova AI on the test set.

Computes SSIM, PSNR, MAE, RMSE, SAM, correlation coefficient,
and optionally LPIPS on every sample in the test split and saves a results CSV.

Usage:
    python evaluate.py
    python evaluate.py --split test
    python evaluate.py --split val --checkpoint checkpoints/best/pix2pix_landsat_best.pth
    python evaluate.py --help
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.landsat9_dataset import Landsat9Dataset
from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_torch_checkpoint


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(max_val**2 / mse))


def compute_ssim_simple(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute a simple SSIM approximation.

    For a more accurate implementation, use torchmetrics or skimage.
    This serves as a fallback when those are unavailable.
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu_pred = np.mean(pred)
    mu_target = np.mean(target)
    sigma_pred = np.std(pred)
    sigma_target = np.std(target)
    sigma_cross = np.mean((pred - mu_pred) * (target - mu_target))

    numerator = (2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)
    denominator = (mu_pred**2 + mu_target**2 + C1) * (sigma_pred**2 + sigma_target**2 + C2)

    return float(numerator / denominator)


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Root Mean Square Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_sam(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Spectral Angle Mapper (in degrees).

    Measures the angular distance between spectral vectors at each pixel.
    Lower is better (0 = identical spectra).
    """
    # pred/target shape: (H, W, C)
    dot = np.sum(pred * target, axis=-1)
    norm_pred = np.linalg.norm(pred, axis=-1)
    norm_target = np.linalg.norm(target, axis=-1)
    denom = norm_pred * norm_target
    # Avoid division by zero
    valid = denom > 1e-8
    if not valid.any():
        return 0.0
    cos_angle = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    angles = np.arccos(cos_angle)
    return float(np.degrees(np.mean(angles)))


def compute_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between flattened arrays."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    if pred_flat.std() == 0 or target_flat.std() == 0:
        return 0.0
    return float(np.corrcoef(pred_flat, target_flat)[0, 1])


@torch.inference_mode()
def evaluate(
    checkpoint_path: str,
    split: str = "test",
    data_root: str = "data/landsat9/splits",
    image_size: int = 256,
    output_csv: str = "outputs/evaluation_results.csv",
    use_lpips: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset split.

    Returns:
        Dictionary with mean metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = load_torch_checkpoint(checkpoint_path, map_location=device)

    model = Pix2Pix(in_channels=1, out_channels=3)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    print(f"Loading {split} split from {data_root}")
    dataset = Landsat9Dataset(
        root_dir=data_root,
        split=split,
        image_size=image_size,
        augment=False,
    )
    print(f"Samples: {len(dataset)}")

    # Optional LPIPS
    lpips_fn = None
    if use_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").to(device)
            lpips_fn.eval()
            print("LPIPS: enabled (AlexNet)")
        except ImportError:
            print("LPIPS: skipped (install lpips package)")
        except Exception as exc:
            print(f"LPIPS: skipped ({exc})")

    # Evaluate
    results: List[Dict[str, float]] = []
    start_time = time.perf_counter()

    try:
        from tqdm import tqdm
        iterator = tqdm(range(len(dataset)), desc="Evaluating", unit="sample")
    except ImportError:
        iterator = range(len(dataset))

    for idx in iterator:
        sample = dataset[idx]
        ir_input = sample["ir"].unsqueeze(0).to(device)
        rgb_target = sample["rgb"].unsqueeze(0).to(device)

        # Generate prediction
        rgb_pred = model.generate(ir_input)

        # Clamp to [0, 1]
        rgb_pred = torch.clamp(rgb_pred, 0.0, 1.0)
        rgb_target = torch.clamp(rgb_target, 0.0, 1.0)

        # Convert to numpy
        pred_np = rgb_pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        target_np = rgb_target.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Compute metrics
        row: Dict[str, float] = {
            "sample_idx": idx,
            "psnr": compute_psnr(pred_np, target_np),
            "ssim": compute_ssim_simple(pred_np, target_np),
            "mae": compute_mae(pred_np, target_np),
            "rmse": compute_rmse(pred_np, target_np),
            "sam": compute_sam(pred_np, target_np),
            "correlation": compute_correlation(pred_np, target_np),
        }

        if lpips_fn is not None:
            # LPIPS expects [-1, 1] range
            lpips_pred = rgb_pred * 2.0 - 1.0
            lpips_target = rgb_target * 2.0 - 1.0
            row["lpips"] = float(lpips_fn(lpips_pred, lpips_target).item())

        results.append(row)

    elapsed = time.perf_counter() - start_time

    # Aggregate
    metric_keys = ["psnr", "ssim", "mae", "rmse", "sam", "correlation"]
    if lpips_fn is not None:
        metric_keys.append("lpips")

    summary: Dict[str, float] = {}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results ({split} split, {len(results)} samples)")
    print(f"{'=' * 60}")
    for key in metric_keys:
        mean = summary.get(f"{key}_mean", 0.0)
        std = summary.get(f"{key}_std", 0.0)
        print(f"  {key.upper():>6s}: {mean:.4f} ± {std:.4f}")
    print(f"  {'TIME':>6s}: {elapsed:.1f}s ({elapsed / max(len(results), 1):.2f}s/sample)")

    # Save CSV
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-sample results saved to: {csv_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate InfraNova AI on the test set."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best/pix2pix_landsat_best.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        default="data/landsat9/splits",
        help="Root directory for dataset splits.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for evaluation (default: 256).",
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluation_results.csv",
        help="Path for the results CSV.",
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Skip LPIPS computation.",
    )
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    evaluate(
        checkpoint_path=args.checkpoint,
        split=args.split,
        data_root=args.data_root,
        image_size=args.image_size,
        output_csv=args.output,
        use_lpips=not args.no_lpips,
    )


if __name__ == "__main__":
    main()
