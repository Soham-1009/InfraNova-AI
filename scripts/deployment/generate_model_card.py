"""
Auto-generate a MODEL_CARD.md for InfraNova AI.

Reads checkpoint metadata and experiment_info.json to produce
a Hugging Face-style model card.

Usage:
    python generate_model_card.py
    python generate_model_card.py --checkpoint checkpoints/best/pix2pix_landsat_best.pth
    python generate_model_card.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_torch_checkpoint


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def generate_model_card(
    checkpoint_path: str = "outputs/best/pix2pix_landsat_best.pth",
    experiment_json: str = "outputs/final/experiment.json",
    output_path: str = "MODEL_CARD.md",
) -> str:
    """Generate and save a MODEL_CARD.md."""

    # Load experiment info if available
    exp_info: dict[str, Any] = {}
    exp_path = Path(experiment_json)
    if exp_path.exists():
        with suppress(Exception):
            exp_info = json.loads(exp_path.read_text(encoding="utf-8"))

    # Load checkpoint info
    ckpt_info: dict[str, Any] = {}
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        with suppress(Exception):
            ckpt = load_torch_checkpoint(checkpoint_path, map_location="cpu")
            ckpt_info = ckpt.get("arch_info", {})
            ckpt_info["epoch"] = ckpt.get("epoch", "unknown")
            ckpt_info["metrics"] = ckpt.get("metrics", {})

    # Instantiate model for parameter counting
    in_channels = int(ckpt_info.get("input_channels", 2))
    out_channels = int(ckpt_info.get("output_channels", 3))
    image_size = int(ckpt_info.get("image_size", 128))
    multi_scale = bool(ckpt_info.get("multi_scale_disc", True))
    num_scales = int(ckpt_info.get("discriminator_scales", 2))

    model = Pix2Pix(
        device="cpu",
        in_channels=in_channels,
        out_channels=out_channels,
        image_size=image_size,
        multi_scale=multi_scale,
        generator_impl=ckpt_info.get("generator_impl", "hd"),
        num_scales=num_scales,
    )
    gen_params, disc_params, total_params = model.count_parameters()

    # Training hyperparameters
    training_hp = exp_info.get("training", {})
    optim_hp = training_hp.get("optimizer", {})
    loss_hp = exp_info.get("loss", {})

    # Best metrics
    metrics = ckpt_info.get("metrics", {})
    best_ssim = metrics.get("val_ssim", exp_info.get("best_ssim", "N/A"))
    best_psnr = metrics.get("val_psnr", exp_info.get("best_psnr", "N/A"))

    card = f"""---
language:
  - en
tags:
  - earth-observation
  - thermal-infrared
  - colorization
  - landsat-9
  - pix2pixhd
  - remote-sensing
---

# InfraNova AI — Dual-Band Thermal IR to RGB Synthesis

## Model Description

**InfraNova AI** is a Pix2PixHD GAN trained to synthesize plausible RGB-like optical images
from Landsat 9 Band 10 + Band 11 dual-band thermal infrared (TIR) data. The model takes
2-channel thermal input and generates a 3-channel RGB visual interpretation at 128x128 resolution.

| Property | Value |
|----------|-------|
| Architecture | Pix2PixHD (GlobalGenerator + LocalEnhancer + MultiScaleDiscriminator) |
| Generator | {ckpt_info.get("generator", "Pix2PixHDGenerator")} |
| Discriminator | {ckpt_info.get("discriminator", "MultiScaleDiscriminator")} |
| Input | {in_channels} x {image_size} x {image_size} (Band 10 + Band 11 thermal) |
| Output | {out_channels} x {image_size} x {image_size} (RGB) |
| Total Parameters | {total_params:,} |
| Generator Parameters | {gen_params:,} |
| Discriminator Parameters | {disc_params:,} |

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Epochs | {training_hp.get("epochs", "N/A")} |
| Batch Size | {training_hp.get("batch_size", "N/A")} |
| Learning Rate | {optim_hp.get("lr", "N/A")} |
| Optimizer | Adam (β₁={optim_hp.get("beta1", "N/A")}, β₂={optim_hp.get("beta2", "N/A")}) |
| λ_adv | {loss_hp.get("lambda_adv", "N/A")} |
| λ_L1 | {loss_hp.get("lambda_l1", "N/A")} |
| λ_perceptual | {loss_hp.get("lambda_perc", "N/A")} |
| λ_SSIM | {loss_hp.get("lambda_ssim", "N/A")} |
| AMP | {training_hp.get("amp", "N/A")} |
| Gradient Clip | {training_hp.get("grad_clip", "N/A")} |

## Performance

| Metric | Value |
|--------|-------|
| Best SSIM | {f"{best_ssim:.4f}" if isinstance(best_ssim, float) else best_ssim} |
| Best PSNR | {f"{best_psnr:.2f} dB" if isinstance(best_psnr, float) else best_psnr} |
| Best SSIM Epoch | {exp_info.get("best_ssim_epoch", "N/A")} |
| Best PSNR Epoch | {exp_info.get("best_psnr_epoch", "N/A")} |
| Total Epochs Trained | {exp_info.get("total_epochs_trained", "N/A")} |

## Dataset

- **Source**: Landsat 9 Level-2 Surface Temperature (Band 10)
- **Regions**: Multiple geographic regions
- **Resolution**: TIR at 100m/pixel, RGB at 100m/pixel
- **Preprocessing**: Percentile-based normalization, resized to 256x256

## Intended Use

This model is intended for:
- Visual interpretation of thermal satellite imagery
- Research and educational purposes
- Interactive demonstrations

> **Note**: The output is a *learned, plausible* RGB-like interpretation and should
> **not** be treated as ground truth visible imagery.

## Limitations

- Trained on a limited number of geographic regions
- Performance may degrade on unseen terrain types
- Single-band thermal input limits spectral information
- Seasonal and atmospheric variations may affect quality

## Environment

| Property | Value |
|----------|-------|
| Python | {exp_info.get("python_version", "N/A").split()[0] if exp_info.get("python_version") else "N/A"} |
| PyTorch | {exp_info.get("pytorch_version", "N/A")} |
| CUDA | {exp_info.get("cuda_device", "N/A")} |
| Git Commit | {exp_info.get("git_commit", "N/A")} |

## Citation

```
@misc{{infranNova2026,
  title={{InfraNova AI: Thermal IR to RGB Synthesis}},
  year={{2026}},
}}
```

---
*Generated on {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}*
"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card.strip() + "\n", encoding="utf-8")
    print(f"Model card saved to: {output_path}")
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate MODEL_CARD.md for InfraNova AI.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best/pix2pix_landsat_best.pth",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--experiment-json",
        default="logs/experiment_info.json",
        help="Experiment info JSON path.",
    )
    parser.add_argument(
        "--output",
        default="MODEL_CARD.md",
        help="Output path for the model card.",
    )
    args = parser.parse_args()

    generate_model_card(
        checkpoint_path=args.checkpoint,
        experiment_json=args.experiment_json,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
