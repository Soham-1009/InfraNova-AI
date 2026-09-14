# Legacy Pix2Pix Architecture & Baseline History — InfraNova AI

**Document Version:** 2.1  
**Date:** 2026-09-13  
**Status:** Historical Experimental Documentation (Preserved for Benchmark Comparison)  

---

## 1. Overview of Historical Baseline

Before adopting the active 2-channel Pix2PixHD architecture, InfraNova AI developed and benchmarked a standard 1-channel Pix2Pix model (`Band 10 Thermal IR` $\to$ `RGB Optical`). This baseline established initial cross-spectral synthesis capabilities but suffered from severe desaturation and limited high-frequency detail.

---

## 2. Legacy Model Architecture

### 2.1. Generator: `GeneratorUNet` (`src/models/pix2pix/generator.py`)
- **Structure**: 8-block fixed depth U-Net designed for $256\times 256$ inputs.
- **Input Channels**: 1 (Band 10 Thermal).
- **Output Channels**: 3 (RGB, Tanh in $[-1, 1]$).
- **Normalization**: `InstanceNorm2d` on all encoder and decoder blocks (except the first layer and the bottleneck).
- **Dropout**: Dropout ($p=0.5$) on the first three decoder blocks.
- **Upsampling**: Bilinear interpolation followed by $3\times 3$ convolution.

### 2.2. Discriminator: Single-Scale `PatchDiscriminator` (`src/models/pix2pix/discriminator.py`)
- **Structure**: 70×70 receptive field PatchGAN.
- **Input Channels**: 4 (1 IR + 3 RGB).
- **Normalization**: Spectral normalization on all convolutional layers.

---

## 3. Historical Baseline Performance (100 Epochs)

| Metric | Historical Single-Band Pix2Pix Baseline (Ep 100) | Active 2-Channel Pix2PixHD (Ep 250) | Improvement Delta |
| :--- | :---: | :---: | :---: |
| **Peak PSNR** | 10.82 dB | **11.75 dB** | **+0.93 dB** |
| **Peak SSIM** | 0.1841 | **0.2522** | **+37.0%** |
| **MAE** | 0.2512 | **0.2183** | **-13.1%** |
| **RMSE** | 0.3120 | **0.2741** | **-12.1%** |
| **CIE $\Delta E^*_{ab}$ Error** | 35.42 | **27.61** | **-22.0%** |
| **SAM (Spectral Angle)** | 0.2680 rad | **0.2050 rad** | **-23.5%** |
| **Saturation Ratio Error vs 1.0**| 0.3977 (Ratio: 0.6023) | **0.0033** (Ratio: 1.0033) | **-99.2%** |
| **10-Epoch Saturation $\sigma$** | 0.3632 | **0.0386** | **-89.4%** |

### Key Takeaways from Historical Baseline:
1. **Desaturation Collapse**: The single-band Pix2Pix model suffered from a severe color collapse towards dull brownish/grayish tones (saturation ratio 0.6023, error 0.3977).
2. **Dual-Band Necessity**: Incorporating Band 11 ($11.50\text{--}12.51\,\mu\text{m}$) provided critical differential water vapor radiance, enabling the Pix2PixHD generator to distinguish vegetative moisture from bare soil.
3. **Loss Suite Effectiveness**: Adding explicit Chroma Loss and Multi-Scale Feature Matching eliminated color instability and lifted the saturation ratio to $1.0033$.
4. **Subsequent Progression (Exp1–Exp9)**: While the 250-epoch dual-band Pix2PixHD model achieved 11.75 dB PSNR and 0.2522 SSIM, the subsequent Exp1–Exp9 research cycle culminated in the `Pix2PixHDGlobalResNetGenerator` (Exp9: LSGAN + SaturationLoss, 11.37M params), achieving **13.75 dB PSNR** and **0.4502 SSIM** on the standardized 1,259-sample test benchmark and being promoted to active production at `outputs/best/pix2pix_landsat_best.pth` (see [`docs/experiments/EXPERIMENTS.md`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/docs/experiments/EXPERIMENTS.md)).