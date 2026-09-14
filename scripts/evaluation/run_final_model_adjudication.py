"""
InfraNova-AI: Final Model Adjudication & Full 1,259-Sample Test Benchmark.
Direct Float32 PyTorch Tensor pipeline to YOLOv8 for zero-copy, memory-safe execution.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.landsat9_dataset import Landsat9Dataset
from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_pix2pix_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("adjudication")


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


def compute_laplacian_var_torch(img_tensor: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=gray.dtype,
        device=gray.device,
    ).unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(gray, kernel, padding=1)
    return lap.view(img_tensor.size(0), -1).var(dim=1)


def compute_fft_hf_energy_torch(img_tensor: torch.Tensor) -> torch.Tensor:
    _b, _c, H, W = img_tensor.shape
    f = torch.fft.fft2(img_tensor)
    fshift = torch.fft.fftshift(f, dim=(-2, -1))
    mag = fshift.abs()
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(
        torch.arange(H, device=img_tensor.device),
        torch.arange(W, device=img_tensor.device),
        indexing="ij",
    )
    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = math.sqrt(cx**2 + cy**2)
    hf_mask = (dist > (0.75 * r_max)).unsqueeze(0).unsqueeze(0)
    total_energy = (mag**2).sum(dim=[-2, -1]).clamp_min(1e-8)
    hf_energy = (mag**2 * hf_mask).sum(dim=[-2, -1])
    return (hf_energy / total_energy).mean(dim=1)


def compute_batch_hist_distance(p_np: np.ndarray, t_np: np.ndarray) -> list[float]:
    B = p_np.shape[0]
    dists = []
    for b in range(B):
        h_dist = 0.0
        for c in range(3):
            hp, _ = np.histogram(p_np[b, c].ravel(), bins=64, range=(0.0, 1.0))
            ht, _ = np.histogram(t_np[b, c].ravel(), bins=64, range=(0.0, 1.0))
            hp = hp / (hp.sum() + 1e-8)
            ht = ht / (ht.sum() + 1e-8)
            h_dist += float(np.sum(np.abs(np.cumsum(hp) - np.cumsum(ht)))) / 64.0
        dists.append(h_dist / 3.0)
    return dists


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 1e-8 else 0.0


def match_detections(pred_boxes: list[np.ndarray], gt_boxes: list[np.ndarray], iou_threshold: float = 0.25):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0, 1.0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, 0.0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), 0.0
    matched_gt = set()
    tp = 0
    ious = []
    for pb in pred_boxes:
        best_iou, best_gt_idx = 0.0, -1
        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_gt_idx = iou, i
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return tp, fp, fn, mean_iou


def load_model(ckpt_path: Path, device: torch.device) -> Pix2Pix:
    model, _architecture = load_pix2pix_model(ckpt_path, device=device)
    return model


def compute_ground_truth_detections(test_loader: DataLoader, device: torch.device) -> list[list[np.ndarray]]:
    logger.info("Computing Ground-Truth Optical RGB YOLO detections...")
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.to("cpu")
    all_gt_boxes = []
    for batch in test_loader:
        target = batch["rgb"].to(device)
        t = (target.clamp(-1.0, 1.0) + 1.0) / 2.0
        res_gt = yolo_model(t, device="cpu", imgsz=128, verbose=False, conf=0.15)
        for r in res_gt:
            boxes = [box.xyxy.cpu().numpy()[0] for box in r.boxes]
            all_gt_boxes.append(boxes)
    del yolo_model
    gc.collect()
    logger.info(f"Cached Ground-Truth detections for all {len(all_gt_boxes)} test samples.")
    return all_gt_boxes


def evaluate_single_model(
    model_name: str,
    ckpt_path: Path,
    test_loader: DataLoader,
    all_gt_boxes: list[list[np.ndarray]],
    device: torch.device,
):
    logger.info("\n=======================================================")
    logger.info(f"Evaluating Model: [{model_name}] from {ckpt_path.name}")
    logger.info("=======================================================")
    model = load_model(ckpt_path, device)
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.to("cpu")

    stats = {k: [] for k in ["psnr", "ssim", "mae", "rmse", "sam", "lab_e", "hist_d", "sat_r", "fft_ratio", "lap_ratio"]}
    yolo_counts = {"tp_25": 0, "fp_25": 0, "fn_25": 0, "tp_50": 0, "fp_50": 0, "fn_50": 0, "total_det": 0, "ious": []}

    sample_offset = 0
    for batch in test_loader:
        ir = batch["ir"].to(device)
        target = batch["rgb"].to(device)
        B = ir.size(0)

        t = (target.clamp(-1.0, 1.0) + 1.0) / 2.0
        lab_t = srgb_to_lab(t)
        fft_t = compute_fft_hf_energy_torch(t)
        lap_t = compute_laplacian_var_torch(t)

        with torch.inference_mode():
            out = model.generate(ir)
            p = (out.clamp(-1.0, 1.0) + 1.0) / 2.0

        mse = F.mse_loss(p, t, reduction="none").mean(dim=[1, 2, 3])
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-8))
        mae = F.l1_loss(p, t, reduction="none").mean(dim=[1, 2, 3])
        rmse = mse.sqrt()

        C1, C2 = 0.01**2, 0.03**2
        mu_p = p.mean(dim=[2, 3], keepdim=True)
        mu_t = t.mean(dim=[2, 3], keepdim=True)
        sigma_p = p.std(dim=[2, 3], keepdim=True)
        sigma_t = t.std(dim=[2, 3], keepdim=True)
        sigma_pt = ((p - mu_p) * (t - mu_t)).mean(dim=[2, 3], keepdim=True)
        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2))
        ssim = ssim.mean(dim=[1, 2, 3])

        dot = (p * t).sum(dim=1)
        norm_p = torch.linalg.norm(p, dim=1)
        norm_t = torch.linalg.norm(t, dim=1)
        cos_a = (dot / (norm_p * norm_t).clamp_min(1e-8)).clamp(-1.0, 1.0)
        sam = torch.acos(cos_a).mean(dim=[1, 2])

        sat_p = p.std(dim=1).mean(dim=[1, 2])
        sat_t = t.std(dim=1).mean(dim=[1, 2])
        sat_ratio = sat_p / sat_t.clamp_min(1e-8)

        lab_p = srgb_to_lab(p)
        delta_e = torch.linalg.norm(lab_p - lab_t, dim=1).mean(dim=[1, 2])

        fft_p = compute_fft_hf_energy_torch(p)
        fft_ratio = fft_p / (fft_t + 1e-8)

        lap_p = compute_laplacian_var_torch(p)
        lap_ratio = lap_p / (lap_t + 1e-8)

        stats["psnr"].extend(psnr.cpu().tolist())
        stats["ssim"].extend(ssim.cpu().tolist())
        stats["mae"].extend(mae.cpu().tolist())
        stats["rmse"].extend(rmse.cpu().tolist())
        stats["sam"].extend(sam.cpu().tolist())
        stats["sat_r"].extend(sat_ratio.cpu().tolist())
        stats["lab_e"].extend(delta_e.cpu().tolist())
        stats["fft_ratio"].extend(fft_ratio.cpu().tolist())
        stats["lap_ratio"].extend(lap_ratio.cpu().tolist())

        p_np = p.cpu().numpy()
        t_np = t.cpu().numpy()
        stats["hist_d"].extend(compute_batch_hist_distance(p_np, t_np))

        # Direct Tensor inference to YOLOv8
        res_p = yolo_model(p, device="cpu", imgsz=128, verbose=False, conf=0.15)

        for b in range(B):
            gt_boxes = all_gt_boxes[sample_offset + b]
            p_boxes = [box.xyxy.cpu().numpy()[0] for box in res_p[b].boxes]

            tp25, fp25, fn25, iou = match_detections(p_boxes, gt_boxes, iou_threshold=0.25)
            tp50, fp50, fn50, _ = match_detections(p_boxes, gt_boxes, iou_threshold=0.50)
            yolo_counts["tp_25"] += tp25
            yolo_counts["fp_25"] += fp25
            yolo_counts["fn_25"] += fn25
            yolo_counts["tp_50"] += tp50
            yolo_counts["fp_50"] += fp50
            yolo_counts["fn_50"] += fn50
            yolo_counts["total_det"] += len(p_boxes)
            if iou > 0:
                yolo_counts["ious"].append(iou)

        sample_offset += B

    del model
    del yolo_model
    gc.collect()

    sat_r_mean = float(np.mean(stats["sat_r"]))
    summary = {
        "psnr_mean": float(np.mean(stats["psnr"])),
        "psnr_std": float(np.std(stats["psnr"])),
        "ssim_mean": float(np.mean(stats["ssim"])),
        "ssim_std": float(np.std(stats["ssim"])),
        "mae_mean": float(np.mean(stats["mae"])),
        "rmse_mean": float(np.mean(stats["rmse"])),
        "sam_mean_rad": float(np.mean(stats["sam"])),
        "sam_mean_deg": float(np.degrees(np.mean(stats["sam"]))),
        "lab_error_mean": float(np.mean(stats["lab_e"])),
        "hist_dist_mean": float(np.mean(stats["hist_d"])),
        "sat_ratio_mean": sat_r_mean,
        "sat_ratio_error": abs(sat_r_mean - 1.0),
        "fft_hf_ratio_mean": float(np.mean(stats["fft_ratio"])),
        "laplacian_var_ratio_mean": float(np.mean(stats["lap_ratio"])),
    }

    tp25, fp25, fn25 = yolo_counts["tp_25"], yolo_counts["fp_25"], yolo_counts["fn_25"]
    prec25 = tp25 / max(tp25 + fp25, 1)
    rec25 = tp25 / max(tp25 + fn25, 1)
    f1_25 = 2 * prec25 * rec25 / max(prec25 + rec25, 1e-8)

    tp50, fp50, fn50 = yolo_counts["tp_50"], yolo_counts["fp_50"], yolo_counts["fn_50"]
    prec50 = tp50 / max(tp50 + fp50, 1)
    rec50 = tp50 / max(tp50 + fn50, 1)
    f1_50 = 2 * prec50 * rec50 / max(prec50 + rec50, 1e-8)

    mean_iou = float(np.mean(yolo_counts["ious"])) if yolo_counts["ious"] else 0.0
    yolo_res = {
        "total_detections": yolo_counts["total_det"],
        "iou_0.25": {
            "tp": tp25,
            "fp": fp25,
            "fn": fn25,
            "precision": float(prec25),
            "recall": float(rec25),
            "f1": float(f1_25),
        },
        "iou_0.50": {
            "tp": tp50,
            "fp": fp50,
            "fn": fn50,
            "precision": float(prec50),
            "recall": float(rec50),
            "f1": float(f1_50),
        },
        "mean_matched_iou": mean_iou,
    }
    logger.info(f"Done [{model_name}]: SSIM={summary['ssim_mean']:.4f}, PSNR={summary['psnr_mean']:.2f}dB, YOLO F1={f1_25:.4f}")
    return summary, yolo_res


def evaluate_thermal_baseline(test_loader: DataLoader, all_gt_boxes: list[list[np.ndarray]], device: torch.device):
    logger.info("\n=======================================================")
    logger.info("Evaluating Thermal False-Color Baseline")
    logger.info("=======================================================")
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.to("cpu")
    yolo_counts = {"tp_25": 0, "fp_25": 0, "fn_25": 0, "tp_50": 0, "fp_50": 0, "fn_50": 0, "total_det": 0, "ious": []}
    sample_offset = 0
    for batch in test_loader:
        ir = batch["ir"].to(device)
        B = ir.size(0)
        b10 = (ir[:, 0:1] + 1.0) / 2.0
        b11 = (ir[:, 1:2] + 1.0) / 2.0
        th_tensor = torch.cat([b10, b11, (b10 + b11) / 2.0], dim=1)

        res_th = yolo_model(th_tensor, device="cpu", imgsz=128, verbose=False, conf=0.15)
        for b in range(B):
            gt_boxes = all_gt_boxes[sample_offset + b]
            th_boxes = [box.xyxy.cpu().numpy()[0] for box in res_th[b].boxes]

            tp25, fp25, fn25, iou = match_detections(th_boxes, gt_boxes, iou_threshold=0.25)
            tp50, fp50, fn50, _ = match_detections(th_boxes, gt_boxes, iou_threshold=0.50)
            yolo_counts["tp_25"] += tp25
            yolo_counts["fp_25"] += fp25
            yolo_counts["fn_25"] += fn25
            yolo_counts["tp_50"] += tp50
            yolo_counts["fp_50"] += fp50
            yolo_counts["fn_50"] += fn50
            yolo_counts["total_det"] += len(th_boxes)
            if iou > 0:
                yolo_counts["ious"].append(iou)
        sample_offset += B

    del yolo_model
    gc.collect()

    tp25, fp25, fn25 = yolo_counts["tp_25"], yolo_counts["fp_25"], yolo_counts["fn_25"]
    prec25 = tp25 / max(tp25 + fp25, 1)
    rec25 = tp25 / max(tp25 + fn25, 1)
    f1_25 = 2 * prec25 * rec25 / max(prec25 + rec25, 1e-8)

    tp50, fp50, fn50 = yolo_counts["tp_50"], yolo_counts["fp_50"], yolo_counts["fn_50"]
    prec50 = tp50 / max(tp50 + fp50, 1)
    rec50 = tp50 / max(tp50 + fn50, 1)
    f1_50 = 2 * prec50 * rec50 / max(prec50 + rec50, 1e-8)

    mean_iou = float(np.mean(yolo_counts["ious"])) if yolo_counts["ious"] else 0.0
    return {
        "total_detections": yolo_counts["total_det"],
        "iou_0.25": {
            "tp": tp25,
            "fp": fp25,
            "fn": fn25,
            "precision": float(prec25),
            "recall": float(rec25),
            "f1": float(f1_25),
        },
        "iou_0.50": {
            "tp": tp50,
            "fp": fp50,
            "fn": fn50,
            "precision": float(prec50),
            "recall": float(rec50),
            "f1": float(f1_50),
        },
        "mean_matched_iou": mean_iou,
    }


def main():
    device = torch.device("cpu")
    logger.info("Initializing Pure-CPU Zero-Copy Adjudication Benchmark...")

    test_ds = Landsat9Dataset(
        root_dir=str(PROJECT_ROOT / "data" / "landsat9_b10_b11" / "splits"),
        split="test",
        image_size=128,
        input_channels=2,
        normalization="local",
        augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    num_samples = len(test_ds)
    logger.info(f"Loaded {num_samples} test samples.")

    # 1. Compute & cache Ground Truth detections once
    all_gt_boxes = compute_ground_truth_detections(test_loader, device)

    # 2. Evaluate Thermal Baseline
    yolo_results = {}
    yolo_results["thermal_baseline"] = evaluate_thermal_baseline(test_loader, all_gt_boxes, device)

    # 3. Evaluate candidate models
    candidates = {
        "exp1_best": PROJECT_ROOT / "outputs" / "exp1" / "best" / "pix2pix_landsat_best.pth",
        "exp2_best": PROJECT_ROOT / "outputs" / "exp2" / "best" / "pix2pix_landsat_best.pth",
        "exp3_best": PROJECT_ROOT / "outputs" / "exp3" / "best" / "pix2pix_landsat_best.pth",
    }

    structural_spectral_results = {}
    for name, ckpt_path in candidates.items():
        summary, y_res = evaluate_single_model(name, ckpt_path, test_loader, all_gt_boxes, device)
        structural_spectral_results[name] = summary
        yolo_results[name] = y_res

    final_report = {
        "test_set_sample_count": num_samples,
        "structural_spectral_metrics": structural_spectral_results,
        "downstream_yolo_benchmark": yolo_results,
    }

    out_file = PROJECT_ROOT / "outputs" / "final_model_adjudication_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("\n=======================================================")
    logger.info(f"Adjudication complete! Saved report to: {out_file}")
    logger.info("=======================================================")


if __name__ == "__main__":
    main()
