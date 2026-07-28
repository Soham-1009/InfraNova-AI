# InfraNova AI Architecture

## Overview
This document outlines the high-level architecture and pipeline of InfraNova AI, a Pix2Pix-based model for translating Landsat 9 thermal infrared (TIR) images into RGB imagery.

## Directory Structure
- `src/`: Core logic and model architectures.
- `scripts/`: Execution scripts broken down by pipeline stage.
- `configs/`: YAML configurations.
- `outputs/`: Generated artifacts and models.

## Pipeline Flow
1. **Download**: `scripts/download/` handles Earth Engine querying and downloading.
2. **Preprocessing**: `scripts/preprocessing/` generates uniform patches and normalizes data.
3. **Training**: `scripts/training/` manages the training loop and ablation studies.
4. **Evaluation**: `scripts/evaluation/` runs PSNR, SSIM, and FID metrics.
5. **Deployment**: `scripts/deployment/` packages models into ONNX/TorchScript.