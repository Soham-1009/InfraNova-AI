"""
Comprehensive Test-Split Evaluation, Candidate Checkpoint Comparison,
and Downstream YOLO Extraction Benchmark (High-Performance Batched Pipeline).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.landsat9_dataset import Landsat9Dataset
from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_torch_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fast PyTorch / Vectorized Metric Computations
# ---------------------------------------------------------------------------
def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, list[float]]:
    """
    Compute all evaluation metrics on a batch of tensors in [-1, 1].
    pred, target: [B, 3, H, W] in [-1, 1]
    """
    # Convert to [0, 1]
    p = (pred.clamp(-1.0, 1.0) + 1.0) / 2.0
    t = (target.clamp(-1.0, 1.0) + 1.0) / 2.0

    B = p.size(0)

    # 1. PSNR & MAE & RMSE
    mse = F.mse_loss(p, t, reduction="none").mean(dim=[1, 2, 3])  # [B]
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-8))  # [B]
    mae = F.l1_loss(p, t, reduction="none").mean(dim=[1, 2, 3])  # [B]
    rmse = mse.sqrt()  # [B]

    # 2. Simple SSIM (per-sample)
    C1 = 0.01**2
    C2 = 0.03**2
    mu_p = p.mean(dim=[2, 3], keepdim=True)
    mu_t = t.mean(dim=[2, 3], keepdim=True)
    sigma_p = p.std(dim=[2, 3], keepdim=True)
    sigma_t = t.std(dim=[2, 3], keepdim=True)
    sigma_pt = ((p - mu_p) * (t - mu_t)).mean(dim=[2, 3], keepdim=True)
    ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2))
    ssim = ssim.mean(dim=[1, 2, 3])  # [B]

    # 3. Spectral Angle Mapper (SAM in radians)
    # Spectral vector at each pixel: dot product across channels (dim=1)
    dot = (p * t).sum(dim=1)  # [B, H, W]
    norm_p = torch.linalg.norm(p, dim=1)  # [B, H, W]
    norm_t = torch.linalg.norm(t, dim=1)  # [B, H, W]
    denom = (norm_p * norm_t).clamp_min(1e-8)
    cos_angle = (dot / denom).clamp(-1.0, 1.0)
    sam = torch.acos(cos_angle).mean(dim=[1, 2])  # [B]

    # 4. Saturation Ratio
    sat_p = p.std(dim=1).mean(dim=[1, 2])  # [B]
    sat_t = t.std(dim=1).mean(dim=[1, 2])  # [B]
    sat_ratio = sat_p / sat_t.clamp_min(1e-8)  # [B]

    # 5. CIE Lab Error (Simplified sRGB -> Lab approximation)
    def srgb_to_lab(img: torch.Tensor) -> torch.Tensor:
        lin = torch.where(img > 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
        r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
        x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
        y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 1.08883
        z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

        def f(c):
            return torch.where(c > 0.008856, c.clamp_min(1e-10).pow(1.0 / 3.0), (903.3 * c + 16.0) / 116.0)

        fx, fy, fz = f(x), f(y), f(z)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b_ch = 200.0 * (fy - fz)
        return torch.cat([L, a, b_ch], dim=1)

    lab_p = srgb_to_lab(p)
    lab_t = srgb_to_lab(t)
    delta_e = torch.linalg.norm(lab_p - lab_t, dim=1).mean(dim=[1, 2])  # [B]

    # 6. Fast Histogram Distance (numpy per sample)
    p_np = p.cpu().numpy()
    t_np = t.cpu().numpy()
    hist_dists = []
    for b in range(B):
        h_dist = 0.0
        for c in range(3):
            hp, _ = np.histogram(p_np[b, c].flatten(), bins=64, range=(0.0, 1.0))
            ht, _ = np.histogram(t_np[b, c].flatten(), bins=64, range=(0.0, 1.0))
            hp = hp.astype(np.float64) / max(hp.sum(), 1e-8)
            ht = ht.astype(np.float64) / max(ht.sum(), 1e-8)
            denom_h = np.maximum(hp + ht, 1e-8)
            h_dist += np.sum((hp - ht) ** 2 / denom_h)
        hist_dists.append(float(h_dist / 3.0))

    return {
        "psnr": psnr.cpu().tolist(),
        "ssim": ssim.cpu().tolist(),
        "mae": mae.cpu().tolist(),
        "rmse": rmse.cpu().tolist(),
        "sam": sam.cpu().tolist(),
        "sat_ratio": sat_ratio.cpu().tolist(),
        "lab_error": delta_e.cpu().tolist(),
        "hist_dist": hist_dists,
    }


def load_candidate_model(checkpoint_path: Path, device: torch.device) -> Pix2Pix:
    logger.info(f"Loading checkpoint: {checkpoint_path.name}")
    ckpt = load_torch_checkpoint(str(checkpoint_path), map_location=device)

    model = Pix2Pix(
        device=device,
        in_channels=2,
        out_channels=3,
        image_size=128,
        generator_impl="hd",
        multi_scale=True,
        num_scales=2,
    )

    if isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
        clean_gen = {k.replace("module.", ""): v for k, v in ckpt["generator_state_dict"].items()}
        model.generator.load_state_dict(clean_gen, strict=True)
        model.eval()
        return model
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model_dict = model.state_dict()
    matched_dict = {k: v for k, v in clean_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(matched_dict, strict=False)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    data_root = PROJECT_ROOT / "data" / "landsat9_b10_b11" / "splits"
    if not data_root.exists():
        logger.error(f"Dataset root not found: {data_root}")
        sys.exit(1)

    test_dataset = Landsat9Dataset(
        root_dir=str(data_root),
        split="test",
        image_size=128,
        input_channels=2,
        augment=False,
    )
    num_test_samples = len(test_dataset)
    logger.info(f"Loaded {num_test_samples} test split samples.")

    logger.info("Pre-loading test split tensors into memory for instant evaluation...")
    all_ir = []
    all_rgb = []
    t0 = time.perf_counter()
    for s in test_dataset:
        all_ir.append(s["ir"])
        all_rgb.append(s["rgb"])
    test_ir_tensor = torch.stack(all_ir, dim=0)  # [1259, 2, 128, 128]
    test_rgb_tensor = torch.stack(all_rgb, dim=0)  # [1259, 3, 128, 128]
    logger.info(
        f"Pre-loaded {num_test_samples} test tensors ({test_ir_tensor.element_size() * test_ir_tensor.nelement() / 1e6:.1f} MB) in {time.perf_counter() - t0:.2f}s"
    )

    candidates_dir = (
        PROJECT_ROOT / "kaggle_kernel" / "run_output" / "outputs" / "pix2pixhd_band10_band11" / "checkpoints"
    )
    final_dir = PROJECT_ROOT / "outputs" / "final"

    candidate_paths = {
        "best_ssim": candidates_dir / "best_ssim.pth",
        "best_psnr": candidates_dir / "best_psnr.pth",
        "best_lab": candidates_dir / "best_lab.pth",
        "best_sam": candidates_dir / "best_sam.pth",
        "best_sat_ratio": candidates_dir / "best_sat_ratio.pth",
        "epoch_250": final_dir / "epoch_250.pth",
    }

    valid_candidates = {}
    for name, path in candidate_paths.items():
        if path.exists():
            valid_candidates[name] = path
        else:
            fallback = final_dir / path.name
            if fallback.exists():
                valid_candidates[name] = fallback
            else:
                logger.warning(f"Checkpoint not found for {name}: {path}")

    logger.info(f"Evaluating {len(valid_candidates)} candidate checkpoints...")

    # =========================================================================
    # Phase 1: Test-Split Quantitative Evaluation (Batched in Memory)
    # =========================================================================
    candidate_metrics = {}
    candidate_models = {}
    batch_size = 64

    for name, path in valid_candidates.items():
        logger.info(f"\nEvaluating Candidate [{name}] on test split...")
        model = load_candidate_model(path, device)
        candidate_models[name] = model

        all_metrics = {
            "psnr": [],
            "ssim": [],
            "mae": [],
            "rmse": [],
            "sam": [],
            "sat_ratio": [],
            "lab_error": [],
            "hist_dist": [],
        }

        start_time = time.perf_counter()

        with torch.inference_mode():
            for i in range(0, num_test_samples, batch_size):
                ir = test_ir_tensor[i : i + batch_size].to(device)
                target = test_rgb_tensor[i : i + batch_size].to(device)

                pred = model.generate(ir)
                batch_res = compute_batch_metrics(pred, target)

                for k in all_metrics:
                    all_metrics[k].extend(batch_res[k])

        elapsed = time.perf_counter() - start_time
        logger.info(f"Finished {name} in {elapsed:.2f}s ({elapsed / num_test_samples * 1000:.2f}ms/sample)")

        summary = {
            "checkpoint_name": name,
            "checkpoint_path": str(path.relative_to(PROJECT_ROOT)),
            "num_test_samples": num_test_samples,
            "eval_duration_sec": float(elapsed),
        }
        for k, vals in all_metrics.items():
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))
            summary[f"{k}_median"] = float(np.median(vals))

        summary["sat_ratio_error"] = float(abs(summary["sat_ratio_mean"] - 1.0))
        candidate_metrics[name] = summary

        print("\n========================================================")
        print(f"Test Split Results for [{name}]:")
        print(f"  PSNR (dB):  {summary['psnr_mean']:.3f} ± {summary['psnr_std']:.3f} dB")
        print(f"  SSIM:       {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
        print(f"  MAE:        {summary['mae_mean']:.4f}")
        print(f"  RMSE:       {summary['rmse_mean']:.4f}")
        print(f"  SAM (rad):  {summary['sam_mean']:.4f} ({np.degrees(summary['sam_mean']):.2f}°)")
        print(f"  CIE Lab:    {summary['lab_error_mean']:.3f}")
        print(f"  Sat Ratio:  {summary['sat_ratio_mean']:.4f} (Error vs 1.0: {summary['sat_ratio_error']:.4f})")
        print(f"  Hist Dist:  {summary['hist_dist_mean']:.4f}")
        print("========================================================")

    # Save to JSON
    out_eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    out_eval_dir.mkdir(parents=True, exist_ok=True)
    out_metrics_file = out_eval_dir / "test_split_candidates_metrics.json"
    with open(out_metrics_file, "w") as f:
        json.dump(candidate_metrics, f, indent=2)
    logger.info(f"Saved test split metrics to {out_metrics_file}")

    # =========================================================================
    # Phase 2: Visual Comparison Grid
    # =========================================================================
    logger.info("\nGenerating Visual Comparison Grid across Candidate Models...")
    num_vis_samples = 4
    vis_indices = [int(i) for i in np.linspace(10, num_test_samples - 10, num_vis_samples)]

    _fig, axes = plt.subplots(num_vis_samples, len(valid_candidates) + 2, figsize=(22, 3.4 * num_vis_samples), dpi=300)

    for row_idx, sample_idx in enumerate(vis_indices):
        sample = test_dataset[sample_idx]
        ir = sample["ir"].unsqueeze(0).to(device)
        target = sample["rgb"].unsqueeze(0).to(device)

        # 1. Thermal False Color
        b10 = (ir[0, 0].cpu().numpy() + 1.0) / 2.0
        b11 = (ir[0, 1].cpu().numpy() + 1.0) / 2.0
        thermal_rgb = np.stack([b10, b11, (b10 + b11) / 2.0], axis=-1)

        ax = axes[row_idx, 0]
        ax.imshow(np.clip(thermal_rgb, 0, 1))
        if row_idx == 0:
            ax.set_title("Input Thermal\n(B10/B11)", fontweight="bold")
        ax.set_ylabel(f"Test Sample #{sample_idx}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # 2. Predictions
        for col_idx, (cand_name, model) in enumerate(candidate_models.items()):
            with torch.inference_mode():
                pred = model.generate(ir)
            pred_np = (pred.clamp(-1.0, 1.0) + 1.0) / 2.0
            pred_np = pred_np.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            ax = axes[row_idx, col_idx + 1]
            ax.imshow(np.clip(pred_np, 0, 1))
            if row_idx == 0:
                ax.set_title(f"Model:\n{cand_name}", fontweight="bold")
            ax.axis("off")

        # 3. Ground Truth
        target_np = (target.clamp(-1.0, 1.0) + 1.0) / 2.0
        target_np = target_np.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ax = axes[row_idx, len(valid_candidates) + 1]
        ax.imshow(np.clip(target_np, 0, 1))
        if row_idx == 0:
            ax.set_title("Ground Truth\nOptical RGB", fontweight="bold", color="#059669")
        ax.axis("off")

    plt.tight_layout()
    vis_fig_path = PROJECT_ROOT / "reports" / "figures" / "candidate_checkpoints_visual_comparison.png"
    vis_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(vis_fig_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved candidate visual comparison figure to {vis_fig_path}")

    # Copy to artifact directory
    artifact_dir = Path(r"C:\Users\soham\.gemini\antigravity-ide\brain\302d9574-41d4-46f0-9917-57def66f3c33")
    if artifact_dir.exists():
        shutil.copy(vis_fig_path, artifact_dir / "candidate_checkpoints_visual_comparison.png")

    # =========================================================================
    # Phase 3: Downstream YOLO Object & Feature Extraction Objective
    # =========================================================================
    logger.info("\nRunning Downstream YOLO Feature & Extraction Benchmark...")
    try:
        from ultralytics import YOLO

        yolo_model = YOLO("yolov8n.pt")

        benchmark_subset_size = min(100, num_test_samples)
        detector_stats = {
            "thermal_input": {"total_detections": 0, "avg_confidence": 0.0},
            "ground_truth_rgb": {"total_detections": 0, "avg_confidence": 0.0},
        }
        for name in valid_candidates:
            detector_stats[name] = {"total_detections": 0, "avg_confidence": 0.0}

        conf_lists = {k: [] for k in detector_stats}

        for idx in range(benchmark_subset_size):
            sample = test_dataset[idx]
            ir = sample["ir"].unsqueeze(0).to(device)
            target = sample["rgb"].unsqueeze(0).to(device)

            b10 = (ir[0, 0].cpu().numpy() + 1.0) / 2.0
            b11 = (ir[0, 1].cpu().numpy() + 1.0) / 2.0
            thermal_img = (np.stack([b10, b11, (b10 + b11) / 2.0], axis=-1) * 255.0).astype(np.uint8)
            target_img = (
                ((target.clamp(-1.0, 1.0) + 1.0) / 2.0).squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
            ).astype(np.uint8)

            res_gt = yolo_model(target_img, verbose=False)[0]
            detector_stats["ground_truth_rgb"]["total_detections"] += len(res_gt.boxes)
            if len(res_gt.boxes) > 0:
                conf_lists["ground_truth_rgb"].extend(res_gt.boxes.conf.cpu().numpy())

            res_th = yolo_model(thermal_img, verbose=False)[0]
            detector_stats["thermal_input"]["total_detections"] += len(res_th.boxes)
            if len(res_th.boxes) > 0:
                conf_lists["thermal_input"].extend(res_th.boxes.conf.cpu().numpy())

            for name, model in candidate_models.items():
                with torch.inference_mode():
                    pred = model.generate(ir)
                pred_img = (
                    ((pred.clamp(-1.0, 1.0) + 1.0) / 2.0).squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
                ).astype(np.uint8)

                res_pred = yolo_model(pred_img, verbose=False)[0]
                detector_stats[name]["total_detections"] += len(res_pred.boxes)
                if len(res_pred.boxes) > 0:
                    conf_lists[name].extend(res_pred.boxes.conf.cpu().numpy())

        for k in detector_stats:
            detector_stats[k]["avg_confidence"] = float(np.mean(conf_lists[k])) if conf_lists[k] else 0.0
            detector_stats[k]["detection_rate_per_patch"] = float(
                detector_stats[k]["total_detections"] / benchmark_subset_size
            )

        logger.info("\nDownstream YOLO Benchmark Results (across 100 test patches):")
        for k, v in detector_stats.items():
            print(
                f"  {k:18s}: Total Detections = {v['total_detections']:3d}, Avg Conf = {v['avg_confidence']:.3f}, Rate/Patch = {v['detection_rate_per_patch']:.2f}"
            )

        out_yolo_file = out_eval_dir / "downstream_yolo_benchmark.json"
        with open(out_yolo_file, "w") as f:
            json.dump(detector_stats, f, indent=2)
        logger.info(f"Saved downstream YOLO benchmark to {out_yolo_file}")

    except Exception as exc:
        logger.error(f"Downstream YOLO evaluation failed: {exc}")

    logger.info("\nAll evaluations complete!")


if __name__ == "__main__":
    main()
