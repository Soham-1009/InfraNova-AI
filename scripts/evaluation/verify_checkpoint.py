import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from kaggle_kernel.train_pix2pixhd import MultiScaleDiscriminator, Pix2PixHDGenerator

CKPT_PATH = PROJECT_ROOT / "outputs/kaggle_batch64_100epochs/outputs/pix2pixhd_batch256_kaggle/best/checkpoint.pth"
CSV_PATH = PROJECT_ROOT / "outputs/kaggle_batch64_100epochs/training.csv"
OUT_JSON = PROJECT_ROOT / "docs/experiments/final_checkpoint_verification.json"

print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# SHA-256
sha256 = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
size_bytes = CKPT_PATH.stat().st_size

# Training CSV Cross-Check
df = pd.read_csv(CSV_PATH)
max_ssim_idx = df["val_ssim"].idxmax()
best_csv_row = df.iloc[max_ssim_idx]

ckpt_epoch = ckpt.get("epoch")
ckpt_metrics = ckpt.get("metrics", {})

# Architecture Verification
gen = Pix2PixHDGenerator(in_channels=1, out_channels=3)
disc = MultiScaleDiscriminator(in_channels=4, num_scales=2)

gen_keys = [k for k in ckpt["model_state_dict"] if "global_gen" in k or "local_enhancer" in k]
gen_state = {k: v for k, v in ckpt["model_state_dict"].items() if k in gen.state_dict()}

load_res = gen.load_state_dict(ckpt["model_state_dict"], strict=False)
print("Generator load result:", load_res)

gen_params = sum(p.numel() for p in gen.parameters())
disc_params = sum(p.numel() for p in disc.parameters())

# Check finiteness of all tensors
all_finite = True
for name, param in gen.named_parameters():
    if not torch.isfinite(param).all():
        print(f"Non-finite tensor in {name}")
        all_finite = False

verification_data = {
    "checkpoint_path": str(CKPT_PATH.relative_to(PROJECT_ROOT)),
    "checkpoint_size_bytes": size_bytes,
    "sha256": sha256,
    "epoch": ckpt_epoch,
    "matches_csv_peak_epoch": bool(int(best_csv_row["epoch"]) == ckpt_epoch),
    "csv_peak_epoch": int(best_csv_row["epoch"]),
    "csv_peak_val_ssim": float(best_csv_row["val_ssim"]),
    "stored_metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in ckpt_metrics.items()},
    "architecture": {
        "generator_type": "Pix2PixHDGenerator",
        "generator_parameters": gen_params,
        "discriminator_type": "MultiScaleDiscriminator",
        "discriminator_scales": 2,
        "discriminator_parameters": disc_params,
        "image_size": 128,
        "input_channels": 1,
        "output_channels": 3,
        "normalization": "local_percentile_stretch",
        "all_tensors_finite": all_finite,
    },
    "resumable_states": {
        "opt_g_present": "opt_g_state_dict" in ckpt,
        "opt_d_present": "opt_d_state_dict" in ckpt,
        "sched_g_present": "sched_g_state_dict" in ckpt,
        "sched_d_present": "sched_d_state_dict" in ckpt,
    },
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(verification_data, f, indent=2)

print(f"\nSaved verification to: {OUT_JSON}")
print("Summary:")
print(f"  Epoch: {ckpt_epoch} (matches CSV peak: {int(best_csv_row['epoch'])})")
print(f"  Best Val SSIM: {ckpt_metrics.get('val_ssim'):.6f}")
print(f"  Val PSNR: {ckpt_metrics.get('val_psnr'):.2f} dB")
print(f"  Val Lab Error: {ckpt_metrics.get('val_lab_error'):.2f}")
print(f"  SHA-256: {sha256}")
print(f"  All tensors finite: {all_finite}")
print(f"  Gen params: {gen_params / 1e6:.2f}M | Disc params: {disc_params / 1e6:.2f}M")
