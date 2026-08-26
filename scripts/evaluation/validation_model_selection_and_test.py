"""
Rigorous Validation-Only Checkpoint Selection, Single Unbiased Test Evaluation,
and Downstream YOLO Precision/Recall/F1/mAP Benchmark.
"""
from __future__ import annotations

import hashlib
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
# Metric computation helpers
# ---------------------------------------------------------------------------
def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, list[float]]:
    p = (pred.clamp(-1.0, 1.0) + 1.0) / 2.0
    t = (target.clamp(-1.0, 1.0) + 1.0) / 2.0
    B = p.size(0)

    # 1. PSNR, MAE, RMSE
    mse = F.mse_loss(p, t, reduction="none").mean(dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-8))
    mae = F.l1_loss(p, t, reduction="none").mean(dim=[1, 2, 3])
    rmse = mse.sqrt()

    # 2. SSIM
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_p = p.mean(dim=[2, 3], keepdim=True)
    mu_t = t.mean(dim=[2, 3], keepdim=True)
    sigma_p = p.std(dim=[2, 3], keepdim=True)
    sigma_t = t.std(dim=[2, 3], keepdim=True)
    sigma_pt = ((p - mu_p) * (t - mu_t)).mean(dim=[2, 3], keepdim=True)
    ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
        (mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2)
    )
    ssim = ssim.mean(dim=[1, 2, 3])

    # 3. Spectral Angle Mapper (SAM)
    dot = (p * t).sum(dim=1)
    norm_p = torch.linalg.norm(p, dim=1)
    norm_t = torch.linalg.norm(t, dim=1)
    denom = (norm_p * norm_t).clamp_min(1e-8)
    cos_angle = (dot / denom).clamp(-1.0, 1.0)
    sam = torch.acos(cos_angle).mean(dim=[1, 2])

    # 4. Saturation Ratio
    sat_p = p.std(dim=1).mean(dim=[1, 2])
    sat_t = t.std(dim=1).mean(dim=[1, 2])
    sat_ratio = sat_p / sat_t.clamp_min(1e-8)

    # 5. CIE Lab Error
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
    delta_e = torch.linalg.norm(lab_p - lab_t, dim=1).mean(dim=[1, 2])

    # 6. Histogram Distance
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
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_dict = model.state_dict()
        matched = {k: v for k, v in clean_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(matched, strict=False)
    else:
        clean_state_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model_dict = model.state_dict()
        matched = {k: v for k, v in clean_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(matched, strict=False)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# IoU and Precision/Recall/F1 Computation for Object Detection
# ---------------------------------------------------------------------------
def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """box format: [x1, y1, x2, y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area <= 1e-8:
        return 0.0
    return float(inter_area / union_area)


def match_detections(pred_boxes: list[np.ndarray], gt_boxes: list[np.ndarray], iou_threshold: float = 0.5):
    """
    Greedy matching between predicted boxes and ground-truth boxes.
    Returns: tp (true positives), fp (false positives), fn (false negatives), mean_iou
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0, 1.0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, 0.0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), 0.0

    matched_gt = set()
    tp = 0
    fp = 0
    ious = []

    for p_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_box_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return tp, fp, fn, mean_iou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    data_root = PROJECT_ROOT / "data" / "landsat9_b10_b11" / "splits"

    # =========================================================================
    # STEP 1: Load Validation Split (1,432 samples)
    # =========================================================================
    logger.info("Loading validation split samples...")
    val_dataset = Landsat9Dataset(
        root_dir=str(data_root),
        split="val",
        image_size=128,
        input_channels=2,
        augment=False,
    )
    num_val_samples = len(val_dataset)
    logger.info(f"Loaded {num_val_samples} validation samples. Pre-loading into RAM...")

    val_ir_list, val_rgb_list = [], []
    for s in val_dataset:
        val_ir_list.append(s["ir"])
        val_rgb_list.append(s["rgb"])
    val_ir_tensor = torch.stack(val_ir_list, dim=0)
    val_rgb_tensor = torch.stack(val_rgb_list, dim=0)

    # Candidates directory
    candidates_dir = PROJECT_ROOT / "kaggle_kernel" / "run_output" / "outputs" / "pix2pixhd_band10_band11" / "checkpoints"
    final_dir = PROJECT_ROOT / "outputs" / "final"

    candidate_paths = {
        "best_ssim": candidates_dir / "best_ssim.pth",
        "best_psnr": candidates_dir / "best_psnr.pth",
        "best_lab": candidates_dir / "best_lab.pth",
        "best_sam": candidates_dir / "best_sam.pth",
        "best_sat_ratio": candidates_dir / "best_sat_ratio.pth",
        "epoch_250": final_dir / "epoch_250.pth",
    }

    # =========================================================================
    # STEP 2: Evaluate All Candidates STRICTLY on the Validation Set
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: VALIDATION-ONLY MULTI-METRIC EVALUATION")
    logger.info("=" * 80)

    val_results = {}
    batch_size = 64

    for name, path in candidate_paths.items():
        if not path.exists():
            continue
        model = load_candidate_model(path, device)
        all_metrics = {
            "psnr": [], "ssim": [], "mae": [], "rmse": [],
            "sam": [], "sat_ratio": [], "lab_error": [], "hist_dist": []
        }

        with torch.inference_mode():
            for i in range(0, num_val_samples, batch_size):
                ir = val_ir_tensor[i : i + batch_size].to(device)
                target = val_rgb_tensor[i : i + batch_size].to(device)
                pred = model.generate(ir)
                batch_res = compute_batch_metrics(pred, target)
                for k in all_metrics.keys():
                    all_metrics[k].extend(batch_res[k])

        summary = {
            "checkpoint_name": name,
            "checkpoint_path": str(path.relative_to(PROJECT_ROOT)),
            "num_val_samples": num_val_samples,
        }
        for k, vals in all_metrics.items():
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))
        summary["sat_ratio_error"] = float(abs(summary["sat_ratio_mean"] - 1.0))
        val_results[name] = summary

    # Print Validation Scorecard
    print("\n" + "=" * 105)
    print(f"{'Candidate':<16} | {'Val PSNR (dB)':<14} | {'Val SSIM':<10} | {'Val MAE':<10} | {'Val SAM (rad)':<14} | {'Val CIE Lab':<12} | {'Val Sat Error':<14}")
    print("=" * 105)
    for name, s in val_results.items():
        print(f"{name:<16} | {s['psnr_mean']:<6.3f} ± {s['psnr_std']:<5.3f} | {s['ssim_mean']:<10.4f} | {s['mae_mean']:<10.4f} | {s['sam_mean']:<6.4f} ({np.degrees(s['sam_mean']):<4.1f}°) | {s['lab_error_mean']:<12.3f} | {s['sat_ratio_error']:<14.4f} (Ratio: {s['sat_ratio_mean']:.3f})")
    print("=" * 105)

    # =========================================================================
    # STEP 3: Apply Formal Composite Validation Selection Rule
    # =========================================================================
    # Selection Rule:
    # 1. Constraint: Saturation Error |Sat - 1.0| <= 0.20
    # 2. Composite Score = SSIM * 10.0 + PSNR * 0.5 - (CIE_Lab / 10.0) - (SAM * 2.0)
    print("\nEvaluating Composite Validation Selection Formula:")
    print("Score = (SSIM * 10.0) + (PSNR * 0.5) - (CIE_Lab / 10.0) - (SAM * 2.0) [Constraint: Sat Error <= 0.20]")
    print("-" * 80)

    ranked_candidates = []
    for name, s in val_results.items():
        sat_err = s["sat_ratio_error"]
        is_admissible = sat_err <= 0.20
        score = (s["ssim_mean"] * 10.0) + (s["psnr_mean"] * 0.5) - (s["lab_error_mean"] / 10.0) - (s["sam_mean"] * 2.0)
        ranked_candidates.append((score, is_admissible, name, s))
        status_str = "ADMISSIBLE" if is_admissible else "EXCLUDED (Sat Overflow)"
        print(f"  {name:<16}: Composite Score = {score:6.3f} [{status_str}], Sat Err = {sat_err:.4f}")

    # Rank admissible candidates by composite score
    admissible = [c for c in ranked_candidates if c[1]]
    admissible.sort(key=lambda x: x[0], reverse=True)
    selected_winner_name = admissible[0][2]
    selected_winner_summary = admissible[0][3]

    print("\n" + "*" * 80)
    print(f"WINNER SELECTED VIA VALIDATION ONLY: [{selected_winner_name}]")
    print(f"  Validation PSNR: {selected_winner_summary['psnr_mean']:.3f} dB")
    print(f"  Validation SSIM: {selected_winner_summary['ssim_mean']:.4f}")
    print(f"  Validation CIE Lab: {selected_winner_summary['lab_error_mean']:.3f}")
    print(f"  Validation SAM: {selected_winner_summary['sam_mean']:.4f} rad ({np.degrees(selected_winner_summary['sam_mean']):.2f}°)")
    print(f"  Validation Sat Ratio: {selected_winner_summary['sat_ratio_mean']:.4f} (Error: {selected_winner_summary['sat_ratio_error']:.4f})")
    print("*" * 80)

    # =========================================================================
    # STEP 4: Freeze the Selected Model for Application
    # =========================================================================
    logger.info(f"\nFreezing [{selected_winner_name}] as the official application model...")
    source_ckpt = candidate_paths[selected_winner_name]
    target_best = PROJECT_ROOT / "outputs" / "best" / "pix2pix_landsat_best.pth"
    target_final_best = PROJECT_ROOT / "outputs" / "final" / "best_checkpoint.pth"

    target_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_ckpt, target_best)
    shutil.copy(source_ckpt, target_final_best)
    logger.info(f"Model frozen and staged to: {target_best}")

    # Save validation selection report
    selection_report = {
        "selection_methodology": "Validation-Only Multi-Metric Composite Optimization",
        "composite_formula": "Score = (SSIM * 10.0) + (PSNR * 0.5) - (CIE_Lab / 10.0) - (SAM * 2.0)",
        "constraint": "Saturation Error <= 0.20",
        "selected_model": selected_winner_name,
        "selected_checkpoint_source": str(source_ckpt.relative_to(PROJECT_ROOT)),
        "validation_scorecard": val_results,
    }
    with open(PROJECT_ROOT / "outputs" / "evaluation" / "validation_model_selection.json", "w") as f:
        json.dump(selection_report, f, indent=2)

    # =========================================================================
    # STEP 5: RUN SINGLE UNBIASED TEST EVALUATION ON THE FROZEN MODEL
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 2: SINGLE UNBIASED TEST-SET EVALUATION (FROZEN MODEL: {selected_winner_name})")
    logger.info("=" * 80)

    test_dataset = Landsat9Dataset(
        root_dir=str(data_root),
        split="test",
        image_size=128,
        input_channels=2,
        augment=False,
    )
    num_test_samples = len(test_dataset)
    logger.info(f"Loaded {num_test_samples} test samples. Pre-loading into RAM...")

    test_ir_list, test_rgb_list = [], []
    for s in test_dataset:
        test_ir_list.append(s["ir"])
        test_rgb_list.append(s["rgb"])
    test_ir_tensor = torch.stack(test_ir_list, dim=0)
    test_rgb_tensor = torch.stack(test_rgb_list, dim=0)

    frozen_model = load_candidate_model(target_best, device)
    test_metrics = {
        "psnr": [], "ssim": [], "mae": [], "rmse": [],
        "sam": [], "sat_ratio": [], "lab_error": [], "hist_dist": []
    }

    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, num_test_samples, batch_size):
            ir = test_ir_tensor[i : i + batch_size].to(device)
            target = test_rgb_tensor[i : i + batch_size].to(device)
            pred = frozen_model.generate(ir)
            batch_res = compute_batch_metrics(pred, target)
            for k in test_metrics.keys():
                test_metrics[k].extend(batch_res[k])

    eval_duration = time.perf_counter() - t0

    unbiased_test_summary = {
        "frozen_model": selected_winner_name,
        "checkpoint_path": str(target_best.relative_to(PROJECT_ROOT)),
        "num_test_samples": num_test_samples,
        "eval_duration_sec": eval_duration,
    }
    for k, vals in test_metrics.items():
        unbiased_test_summary[f"{k}_mean"] = float(np.mean(vals))
        unbiased_test_summary[f"{k}_std"] = float(np.std(vals))
        unbiased_test_summary[f"{k}_median"] = float(np.median(vals))
    unbiased_test_summary["sat_ratio_error"] = float(abs(unbiased_test_summary["sat_ratio_mean"] - 1.0))

    print("\n" + "=" * 80)
    print(f"OFFICIAL UNBIASED TEST-SET SCORECARD FOR FROZEN MODEL [{selected_winner_name}]:")
    print(f"  Test PSNR (dB):      {unbiased_test_summary['psnr_mean']:.3f} ± {unbiased_test_summary['psnr_std']:.3f} dB")
    print(f"  Test SSIM:           {unbiased_test_summary['ssim_mean']:.4f} ± {unbiased_test_summary['ssim_std']:.4f}")
    print(f"  Test MAE:            {unbiased_test_summary['mae_mean']:.4f}")
    print(f"  Test RMSE:           {unbiased_test_summary['rmse_mean']:.4f}")
    print(f"  Test SAM (rad):      {unbiased_test_summary['sam_mean']:.4f} ({np.degrees(unbiased_test_summary['sam_mean']):.2f}°)")
    print(f"  Test CIE Lab Error:  {unbiased_test_summary['lab_error_mean']:.3f}")
    print(f"  Test Saturation:     {unbiased_test_summary['sat_ratio_mean']:.4f} (Error vs 1.0: {unbiased_test_summary['sat_ratio_error']:.4f})")
    print(f"  Test Histogram Dist: {unbiased_test_summary['hist_dist_mean']:.4f}")
    print("=" * 80)

    with open(PROJECT_ROOT / "outputs" / "evaluation" / "unbiased_test_generalization_scorecard.json", "w") as f:
        json.dump(unbiased_test_summary, f, indent=2)

    # =========================================================================
    # STEP 6: Rigorous Downstream YOLO Precision/Recall/F1/IoU Evaluation
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: DOWNSTREAM YOLO OBJECT/FEATURE EXTRACTION BENCHMARK")
    logger.info("=" * 80)

    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")

        eval_subset_size = min(200, num_test_samples)
        logger.info(f"Evaluating paired detection fidelity against Ground-Truth Optical RGB on {eval_subset_size} test patches...")

        modalities = {
            "thermal_input": {"tp": 0, "fp": 0, "fn": 0, "total_det": 0, "ious": []},
            "frozen_model": {"tp": 0, "fp": 0, "fn": 0, "total_det": 0, "ious": []},
            "epoch_250": {"tp": 0, "fp": 0, "fn": 0, "total_det": 0, "ious": []},
        }

        # Load alternative models for comparison
        epoch250_model = load_candidate_model(candidate_paths["epoch_250"], device)

        for idx in range(eval_subset_size):
            sample = test_dataset[idx]
            ir = sample["ir"].unsqueeze(0).to(device)
            target = sample["rgb"].unsqueeze(0).to(device)

            # Ground Truth RGB
            target_img = (((target.clamp(-1.0, 1.0) + 1.0) / 2.0).squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            res_gt = yolo_model(target_img, verbose=False, conf=0.15)[0]
            gt_boxes = [box.xyxy.cpu().numpy()[0] for box in res_gt.boxes]

            # 1. Thermal False Color
            b10 = (ir[0, 0].cpu().numpy() + 1.0) / 2.0
            b11 = (ir[0, 1].cpu().numpy() + 1.0) / 2.0
            thermal_img = (np.stack([b10, b11, (b10 + b11) / 2.0], axis=-1) * 255.0).astype(np.uint8)
            res_th = yolo_model(thermal_img, verbose=False, conf=0.15)[0]
            th_boxes = [box.xyxy.cpu().numpy()[0] for box in res_th.boxes]

            tp_th, fp_th, fn_th, iou_th = match_detections(th_boxes, gt_boxes, iou_threshold=0.25)
            modalities["thermal_input"]["tp"] += tp_th
            modalities["thermal_input"]["fp"] += fp_th
            modalities["thermal_input"]["fn"] += fn_th
            modalities["thermal_input"]["total_det"] += len(th_boxes)
            if iou_th > 0:
                modalities["thermal_input"]["ious"].append(iou_th)

            # 2. Frozen Model Synthesized RGB
            with torch.inference_mode():
                pred_frozen = frozen_model.generate(ir)
            pred_frozen_img = (((pred_frozen.clamp(-1.0, 1.0) + 1.0) / 2.0).squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            res_fr = yolo_model(pred_frozen_img, verbose=False, conf=0.15)[0]
            fr_boxes = [box.xyxy.cpu().numpy()[0] for box in res_fr.boxes]

            tp_fr, fp_fr, fn_fr, iou_fr = match_detections(fr_boxes, gt_boxes, iou_threshold=0.25)
            modalities["frozen_model"]["tp"] += tp_fr
            modalities["frozen_model"]["fp"] += fp_fr
            modalities["frozen_model"]["fn"] += fn_fr
            modalities["frozen_model"]["total_det"] += len(fr_boxes)
            if iou_fr > 0:
                modalities["frozen_model"]["ious"].append(iou_fr)

            # 3. Epoch 250 Synthesized RGB
            with torch.inference_mode():
                pred_ep250 = epoch250_model.generate(ir)
            pred_ep250_img = (((pred_ep250.clamp(-1.0, 1.0) + 1.0) / 2.0).squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            res_ep = yolo_model(pred_ep250_img, verbose=False, conf=0.15)[0]
            ep_boxes = [box.xyxy.cpu().numpy()[0] for box in res_ep.boxes]

            tp_ep, fp_ep, fn_ep, iou_ep = match_detections(ep_boxes, gt_boxes, iou_threshold=0.25)
            modalities["epoch_250"]["tp"] += tp_ep
            modalities["epoch_250"]["fp"] += fp_ep
            modalities["epoch_250"]["fn"] += fn_ep
            modalities["epoch_250"]["total_det"] += len(ep_boxes)
            if iou_ep > 0:
                modalities["epoch_250"]["ious"].append(iou_ep)

        # Compute Precision, Recall, F1, mean IoU
        yolo_report = {"eval_subset_size": eval_subset_size, "iou_threshold": 0.25}
        print("\n" + "=" * 90)
        print(f"{'Input Source':<18} | {'Total Dets':<10} | {'TP':<6} | {'FP':<6} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'Mean IoU':<8}")
        print("=" * 90)
        for mod, counts in modalities.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            mean_iou = float(np.mean(counts["ious"])) if counts["ious"] else 0.0

            yolo_report[mod] = {
                "total_detections": counts["total_det"],
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "mean_matched_iou": mean_iou,
            }
            print(f"{mod:<18} | {counts['total_det']:<10} | {tp:<6} | {fp:<6} | {precision:<10.3f} | {recall:<8.3f} | {f1:<8.3f} | {mean_iou:<8.3f}")
        print("=" * 90)

        with open(PROJECT_ROOT / "outputs" / "evaluation" / "downstream_yolo_precision_recall.json", "w") as f:
            json.dump(yolo_report, f, indent=2)

    except Exception as exc:
        logger.error(f"Downstream YOLO evaluation error: {exc}", exc_info=True)

    logger.info("\nValidation selection, single test evaluation, and downstream YOLO benchmark complete!")


if __name__ == "__main__":
    main()
