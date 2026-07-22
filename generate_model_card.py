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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_torch_checkpoint


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def generate_model_card(
    checkpoint_path: str = "checkpoints/best/pix2pix_landsat_best.pth",
    experiment_json: str = "logs/experiment_info.json",
    output_path: str = "MODEL_CARD.md",
) -> str:
    """Generate and save a MODEL_CARD.md."""

    # Load experiment info if available
    exp_info: Dict[str, Any] = {}
    exp_path = Path(experiment_json)
    if exp_path.exists():
        try:
            exp_info = json.loads(exp_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Load checkpoint info
    ckpt_info: Dict[str, Any] = {}
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        try:
            ckpt = load_torch_checkpoint(checkpoint_path, map_location="cpu")
            ckpt_info = ckpt.get("arch_info", {})
            ckpt_info["epoch"] = ckpt.get("epoch", "unknown")
            ckpt_info["metrics"] = ckpt.get("metrics", {})
        except Exception:
            pass

    # Model info
    model = Pix2Pix(in_channels=1, out_channels=3)
    gen_params = _count_params(model.generator)
    disc_params = _count_params(model.discriminator)
    total_params = gen_params + disc_params

    # Hyperparams
    hp = exp_info.get("hyperparameters", {})
    training_hp = hp.get("training", {})
    optim_hp = training_hp.get("optimizer", {})
    loss_hp = training_hp.get("loss", {})

    # Metrics
    metrics = ckpt_info.get("metrics", {})
    best_ssim = exp_info.get("best_ssim", metrics.get("val_ssim", "N/A"))
    best_psnr = exp_info.get("best_psnr", metrics.get("val_psnr", "N/A"))

    card = f"""---
language: en
license: mit
tags:
  - image-to-image
  - thermal-infrared
  - landsat-9
  - pix2pix
  - remote-sensing
  - ISRO
---

# InfraNova AI — Thermal IR to RGB Synthesis

## Model Description

**InfraNova AI** is a Pix2Pix GAN trained to synthesize plausible RGB-like images
from Landsat 9 Band 10 thermal infrared (TIR) data. The model takes single-channel
thermal input and generates a 3-channel RGB visual interpretation.

| Property | Value |
|----------|-------|
| Architecture | Pix2Pix (U-Net Generator + PatchGAN Discriminator) |
| Generator | {ckpt_info.get('generator', 'UNetGenerator')} |
| Discriminator | {ckpt_info.get('discriminator', 'PatchGANDiscriminator')} |
| Input | 1 × 256 × 256 (thermal) |
| Output | 3 × 256 × 256 (RGB) |
| Total Parameters | {total_params:,} |
| Generator Parameters | {gen_params:,} |
| Discriminator Parameters | {disc_params:,} |

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Epochs | {training_hp.get('epochs', 'N/A')} |
| Batch Size | {training_hp.get('batch_size', 'N/A')} |
| Learning Rate | {optim_hp.get('lr', 'N/A')} |
| Optimizer | Adam (β₁={optim_hp.get('beta1', 'N/A')}, β₂={optim_hp.get('beta2', 'N/A')}) |
| λ_adv | {loss_hp.get('lambda_adv', 'N/A')} |
| λ_L1 | {loss_hp.get('lambda_l1', 'N/A')} |
| λ_perceptual | {loss_hp.get('lambda_perc', 'N/A')} |
| λ_SSIM | {loss_hp.get('lambda_ssim', 'N/A')} |
| AMP | {training_hp.get('amp', 'N/A')} |
| Gradient Clip | {training_hp.get('grad_clip', 'N/A')} |

## Performance

| Metric | Value |
|--------|-------|
| Best SSIM | {f'{best_ssim:.4f}' if isinstance(best_ssim, float) else best_ssim} |
| Best PSNR | {f'{best_psnr:.2f} dB' if isinstance(best_psnr, float) else best_psnr} |
| Best SSIM Epoch | {exp_info.get('best_ssim_epoch', 'N/A')} |
| Best PSNR Epoch | {exp_info.get('best_psnr_epoch', 'N/A')} |
| Total Epochs Trained | {exp_info.get('total_epochs_trained', 'N/A')} |

## Dataset

- **Source**: Landsat 9 Level-2 Surface Temperature (Band 10)
- **Regions**: Multiple geographic regions
- **Resolution**: TIR at 100m/pixel, RGB at 100m/pixel
- **Preprocessing**: Percentile-based normalization, resized to 256×256

## Intended Use

This model is intended for:
- Visual interpretation of thermal satellite imagery
- Research and educational purposes
- Hackathon demonstrations

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
| Python | {exp_info.get('python_version', 'N/A').split()[0] if exp_info.get('python_version') else 'N/A'} |
| PyTorch | {exp_info.get('pytorch_version', 'N/A')} |
| CUDA | {exp_info.get('cuda_device', 'N/A')} |
| Git Commit | {exp_info.get('git_commit', 'N/A')} |

## Citation

```
@misc{{infranNova2026,
  title={{InfraNova AI: Thermal IR to RGB Synthesis}},
  year={{2026}},
  note={{Bharatiya Antariksh Hackathon 2026 — ISRO}}
}}
```

---
*Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*
"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card.strip() + "\n", encoding="utf-8")
    print(f"Model card saved to: {output_path}")
    return card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-generate MODEL_CARD.md for InfraNova AI."
    )
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
