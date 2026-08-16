# InfraNova AI — Configuration System

This document outlines the YAML configuration system used to orchestrate model training.

## 1. Structure

The main configuration file is `configs/config.yaml`. The training script `src/training/train_landsat.py` loads this file to define hyperparameters.

### 1.1 Project Settings
- `name`, `task`, `modality`: Metadata used for logging and tracking.
- `seed`: Global random seed (42) to ensure deterministic data splits and initialization.

### 1.2 Dataset Settings
- `image_size`: `128` (Must match the spatial dimensions of `.npy` patches).
- `input_channels`: `1` (TIR).
- `output_channels`: `3` (RGB).
- `normalization`: Mode `local` means per-patch percentile stretching is used instead of global dataset statistics.

### 1.3 Model Settings
- `generator.implementation`: `dynamic` (Uses `GeneratorUNetDynamic`). If changed to `legacy`, it attempts to load the old 256x256 static U-Net.
- `multi_scale_disc`: `false`. Uses a single PatchGAN.

### 1.4 Training Hyperparameters
- `epochs`: `500`.
- `batch_size`: `256`.
- `decay_start_epoch`: `450` (LR remains constant until epoch 450, then decays to 0 by 500).
- `patience`: `100` (Early stopping threshold).
- `amp`: `true` (Automatic Mixed Precision).
- `grad_clip`: `1.0`.

### 1.5 Loss Weights
- `lambda_adv`: `1.0` (GAN loss).
- `lambda_l1`: `10.0` (Pixel-wise L1 loss).
- `lambda_perc`: `10.0` (VGG Perceptual loss).
- `lambda_ssim`: `5.0` (SSIM structure loss).
- `lambda_chroma`: `2.0` (Chroma/Color loss).
- `lambda_feat`: `0.0` (Feature matching loss - currently disabled).
- `gan_mode`: `bce` (Binary Cross Entropy).

## 2. Config Overrides

The training script supports command-line overrides for any parameter. This is useful for sweep scripts or quick tests.

**Syntax**: `--overrides key=value key2=value2`

**Example**:
```bash
python src/training/train_landsat.py --config configs/config.yaml --overrides epochs=1 batch_size=4 training.amp=false
```

*Note: The `config_smoke.yaml` file mentioned in the README does not exist. Overrides should be used for smoke testing instead.*
