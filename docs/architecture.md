# InfraNova AI Architecture

## Overview
This document outlines the high-level architecture and pipeline of InfraNova AI, a Pix2Pix-based model for translating Landsat 9 thermal infrared (TIR) images into RGB imagery. 

A primary architectural decision is the use of a **Dynamic U-Net Generator**, which computes its bottleneck depth dynamically based on the input spatial dimensions (e.g., 128x128 vs 256x256).

## Directory Structure
- `configs/`: YAML configurations, including `config.yaml` and `config_smoke.yaml`.
- `data/`: Contains raw downloads, processed patches, and train/val/test splits.
- `scripts/`: Execution scripts broken down by pipeline stage.
- `src/`: Core logic, dynamic model architectures, datasets, and utilities.
- `outputs/`: Generated artifacts, checkpoints, and telemetry.

## Pipeline Flow
1. **Download**: `scripts/download/` handles Earth Engine querying and downloading.
2. **Preprocessing**: `scripts/preprocessing/` generates uniform 128x128 patches and performs telemetry-driven filtering (e.g. discarding HIGH_NODATA or ALL_ZERO patches).
3. **Training**: `scripts/training/` manages the robust training loop, featuring deterministic resuming and metric logging.
4. **Evaluation**: `scripts/evaluation/` runs PSNR, SSIM, and FID metrics as well as inference consistency checks.
5. **Deployment**: `scripts/deployment/` handles Kaggle packaging and ONNX/TorchScript export.