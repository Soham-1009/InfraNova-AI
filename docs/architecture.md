# InfraNova AI Architecture

## Overview

InfraNova AI is a Pix2Pix-based model for translating Landsat 9 thermal infrared (TIR) images into RGB imagery. The generator uses a **Dynamic U-Net** whose bottleneck depth is computed automatically from the input spatial dimensions (e.g., 7 encoding layers for 128×128 input).

## Model Components

| Component | File | Description |
|-----------|------|-------------|
| Dynamic U-Net Generator | `src/models/pix2pix/generator_dynamic.py` | Encoder-decoder with skip connections; depth adapts to input size |
| Static U-Net Generator | `src/models/pix2pix/generator.py` | Fixed 8-layer encoder-decoder (legacy) |
| PatchGAN Discriminator | `src/models/pix2pix/discriminator.py` | 70×70 receptive-field patch classifier |
| Pix2Pix Wrapper | `src/models/pix2pix/pix2pix.py` | Combines generator + discriminator into a single module |

The production configuration (`configs/config.yaml`) selects the dynamic generator via `model.generator.implementation: "dynamic"`.

## Loss Functions

Defined in `src/training/losses.py`:

| Loss | Weight | Purpose |
|------|--------|---------|
| Adversarial (BCE) | 1.0 | Encourages realistic textures |
| L1 | 10.0 | Pixel-level structural fidelity |
| Perceptual (VGG) | 10.0 | High-level feature similarity |
| SSIM | 5.0 | Structural similarity |
| Chroma | 2.0 | Color consistency |

## Directory Structure

```text
configs/          YAML configurations (config.yaml, config_smoke.yaml)
data/             Raw downloads, processed patches, train/val/test splits
scripts/          Pipeline scripts by stage (download, preprocessing, evaluation, deployment, training)
src/              Core source code
  datasets/       Landsat 9 dataset loader
  models/         Dynamic U-Net generator, PatchGAN discriminator
  training/       Training loop, losses, callbacks, scheduler
  inference/      Production inference engine
  evaluation/     Metric computation modules
  detection/      Object detection integration
  losses/         (reserved)
  utils/          Checkpoint management, logging, helpers
demo/             Streamlit web demo
notebooks/        Kaggle training notebook (MAIN.ipynb)
tests/            Unit and integration tests
outputs/          Checkpoints, visualizations, telemetry
logs/             Training CSV logs, TensorBoard logs
```

## Pipeline Flow

1. **Download** — `scripts/download/` queries Google Earth Engine and exports Landsat 9 bands.
2. **Preprocessing** — `scripts/preprocessing/` builds 128×128 paired patches, filters anomalous data, and creates region-level train/val/test splits.
3. **Training** — `src/training/train_landsat.py` runs the Pix2Pix training loop with AMP, deterministic checkpointing, and CSV metric logging.
4. **Evaluation** — `scripts/evaluation/` computes PSNR, SSIM, LPIPS, FID, and runs inference consistency tests.
5. **Deployment** — `scripts/deployment/` generates model cards, handles batch inference, and exports to ONNX/TorchScript.

## Training Configuration

Key production settings from `configs/config.yaml`:

| Parameter | Value |
|-----------|-------|
| Image size | 128×128 |
| Epochs | 250 |
| Batch size | 8 |
| LR decay start | Epoch 230 |
| Learning rate | 0.0002 |
| AMP | Enabled |
| Generator | Dynamic U-Net |