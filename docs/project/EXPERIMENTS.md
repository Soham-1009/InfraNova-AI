# InfraNova AI — Experiment History

This document summarizes the training sessions run so far, extracted from `logs/experiment_history.csv` and `logs/experiment_info.json`.

## 1. Environment Details
- **Platform**: Kaggle Notebooks
- **GPU**: 2× NVIDIA Tesla T4
- **PyTorch Version**: 2.10.0+cu128
- **Latest Commit Hash**: `9d9cf9da982dc29bf35a68cba36e02ca99a21c75`

## 2. Session Log

Training was performed across 6 resumable sessions (to work around Kaggle timeout limits). The model automatically loaded the `latest` checkpoint and resumed optimization.

| Session Date | Epochs Ran | Best SSIM | Best SSIM Epoch | Best PSNR | Val Saturation Ratio | Val Lab Error |
|--------------|------------|-----------|-----------------|-----------|----------------------|---------------|
| 2026-07-30 | 1 | 0.3151 | 1 | 11.86 | 1.032 | 28.04 |
| 2026-07-31 | 1 | 0.3381 | 2 | 11.80 | 0.645 | 27.95 |
| 2026-07-31 | 2 | 0.2566 | 1 | 11.49 | 0.759 | 29.57 |
| 2026-07-31 | 1 | 0.3115 | 2 | 11.81 | 0.701 | 28.14 |
| 2026-07-31 | 10 | 0.3673 | 5 | 11.93 | 0.356 | 27.18 |
| **2026-08-09** | **59** | **0.4290** | **226** | **13.27** | **0.438** | **24.02** |

## 3. Current Best Model
- **Epoch**: 226
- **SSIM**: 0.4290
- **PSNR**: 13.27 dB
- **Color Metric**: Val Saturation Ratio is 0.438. This indicates the model is struggling to generate vibrant colors (outputs are roughly half as saturated as the real ground truth). 
- **Location**: `outputs/models/best/pix2pix_landsat_best.pth`

## 4. Next Steps for Experimentation
To push SSIM above 0.43 and fix the desaturation issue, future experiments should test:
1. **Increasing `lambda_chroma`**: Currently at 2.0. Increasing to 5.0 or 10.0 might force better color saturation.
2. **Global Normalization**: Switch dataset `normalization.mode` to `global` to see if per-sample percentile stretching is washing out absolute thermal patterns.
3. **Multi-Scale Discriminator**: Enable `multi_scale_disc: true` to evaluate images at multiple resolutions simultaneously.
