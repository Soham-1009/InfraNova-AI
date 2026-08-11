# InfraNova AI — Training Pipeline

This document details the mechanics of how the Pix2Pix model is trained in this repository.

## 1. Execution Context

Because the training process is computationally heavy, it is designed to run in a Kaggle notebook environment with GPU acceleration (specifically, Dual Tesla T4 GPUs).

- **Entry Point**: `src/training/train_landsat.py`
- **Execution Script**: `notebooks/MAIN.ipynb`
- **Configuration**: `configs/config.yaml`

## 2. Training Loop Mechanics

The core training logic lives in `src/training/trainer.py`.

### 2.1 Hardware Acceleration & Multi-GPU
- **Device Placement**: The `Pix2Pix` wrapper forces all tensors to the specified device.
- **DataParallel**: If multiple GPUs are detected, the model is wrapped in `torch.nn.DataParallel` allowing the batch to be split across GPUs. 
  - *Note: This adds a `.module.` prefix to state dict keys, which is stripped during checkpoint loading in `checkpoint.py`.*
- **AMP (Automatic Mixed Precision)**: Enabled by default (`amp: true`). Gradients are scaled using `torch.cuda.amp.GradScaler` to prevent underflow while training in FP16, resulting in a ~2x speedup on TensorCore architectures.

### 2.2 Forward & Backward Pass
For every batch of shape `[B, 1, 128, 128]` (IR) and `[B, 3, 128, 128]` (RGB):

1. **Discriminator Phase**:
   - Generator creates `fake_rgb` from `ir` inside a `torch.no_grad()` context block (we don't backprop through the generator here).
   - Discriminator is trained on `[ir, real_rgb]` (Target: 1.0) and `[ir, fake_rgb]` (Target: 0.0).
   - Discriminator optimizer steps.
2. **Generator Phase**:
   - Generator creates `fake_rgb` (with gradients enabled).
   - Discriminator evaluates `[ir, fake_rgb]`.
   - `CombinedLoss` calculates GAN, L1, Perceptual, SSIM, and Chroma errors.
   - Generator optimizer steps.

### 2.3 Optimization
- **Optimizer**: Adam
- **Generator LR**: 0.0002
- **Discriminator LR**: 0.0001 (Half the generator LR to prevent the discriminator from winning too quickly)
- **Beta1**: 0.5 (Standard for GANs)
- **Beta2**: 0.999
- **Gradient Clipping**: Clipped to max norm of `1.0` to prevent loss explosions during adversarial instability.

### 2.4 Learning Rate Scheduling
Configured to use a Linear LR Scheduler (`src/training/scheduler.py`).
- **Decay**: Maintains a constant LR until `decay_start_epoch` (230), then linearly decays the LR to 0 over the remaining epochs (up to 250).

## 3. Validation & Metrics

After every epoch, the model is evaluated on the validation split.

### 3.1 Primary Metrics
- **SSIM (Structural Similarity Index)**: Measures structural fidelity. (Higher is better, max 1.0) -> *Used to determine the `best` checkpoint.*
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level error. (Higher is better)

### 3.2 Diagnostic Metrics
- **val_sat_ratio**: Ratio of saturation in generated images vs real images. (1.0 = perfect match, <1.0 = desaturated outputs)
- **val_hist_dist**: Bhattacharyya distance between color histograms. (Lower is better)
- **val_lab_error**: Mean absolute error in CIELAB color space. (Lower is better)

## 4. Checkpoints & Logging

- **Checkpoints**: Saved to `outputs/models/`.
  - `latest/pix2pix_landsat_latest.pth`: Saved every epoch.
  - `best/pix2pix_landsat_best.pth`: Saved only when validation SSIM reaches a new high.
- **Resumption**: The script automatically resumes from `latest_checkpoint` if the file exists, making Kaggle preemptions safe.
- **Logging**: Metrics are appended to `logs/experiment_history.csv` and `logs/training.csv`. (WandB is supported but currently disabled).

## 5. Starting a Training Run

To start training locally (for debugging):
```bash
python src/training/train_landsat.py --config configs/config.yaml
```

To run a smoke test (verifying the loop works without training fully):
```bash
python src/training/train_landsat.py --config configs/config.yaml --overrides epochs=1 batch_size=4
```
