# Model Architecture Specification — InfraNova AI

**Document Version:** 2.0 (Pix2PixHD Dual-Band Specification)  
**Date:** 2026-08-26  
**Status:** Authoritative Model Specification  

---

## 1. Architectural Overview

InfraNova AI employs a **multi-scale conditional Generative Adversarial Network** based on Pix2PixHD, tailored specifically for dual-band thermal radiance input ($128\times 128$) to true color optical RGB synthesis.

```
+-----------------------------------------------------------------------------+
|                                 INPUT TENSOR                                |
|             [B, 2, 128, 128] (Band 10 + Band 11, Range: [-1, 1])            |
+-----------------------------------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v (2x Average Pooling)                  v (Full Resolution)
+------------------------------------+   +------------------------------------+
|          GLOBAL GENERATOR          |   |           LOCAL ENHANCER           |
|      Depth-6 Coarse U-Net          |   |  enc1: Conv4x4 (in=2, out=32)      |
|  Down: 2 -> 64 -> 128 -> 256->512  |   |  enc2: Conv4x4 (in=32, out=64, IN) |
|  Up:   512 -> 256 -> 128           |   |                                    |
|  Global Latent: [B, 128, 64, 64]   |-->|  Fusion: local + 1x1_proj(global)  |
|  Coarse RGB:   [B, 3, 64, 64]      |   |  dec1: UpBilinear (in=64, out=32)  |
+------------------------------------+   |  final: UpBilinear -> 3x3 Conv     |
                                         +------------------------------------+
                                                           |
                                                           v
                                         +------------------------------------+
                                         |         GENERATED RGB IMAGE        |
                                         |     [B, 3, 128, 128], Tanh [-1, 1] |
                                         +------------------------------------+
```

---

## 2. Generator Specification (`Pix2PixHDGenerator`)

**File**: [`src/models/pix2pix/generator_hd.py`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/models/pix2pix/generator_hd.py)  
**Total Parameters**: **21,382,915** (21.38M)  
**Input Shape**: `[B, 2, 128, 128]`  
**Output Shape**: `[B, 3, 128, 128]` (Range $[-1, 1]$ via Tanh)  

### 2.1. GlobalGenerator (Coarse Network)
- **Input**: Downsampled thermal tensor $[B, 2, 64, 64]$ via `F.avg_pool2d(kernel_size=2, stride=2)`.
- **Encoder**:
  - `down0`: `Conv2d(2, 64, kernel=4, stride=2, pad=1)` $\to$ `LeakyReLU(0.2)`
  - `down1`: `Conv2d(64, 128, kernel=4, stride=2, pad=1)` $\to$ `InstanceNorm` $\to$ `LeakyReLU(0.2)`
  - `down2`: `Conv2d(128, 256, kernel=4, stride=2, pad=1)` $\to$ `InstanceNorm` $\to$ `LeakyReLU(0.2)`
  - `down3`: `Conv2d(256, 512, kernel=4, stride=2, pad=1)` $\to$ `InstanceNorm` $\to$ `LeakyReLU(0.2)`
  - `down4`: `Conv2d(512, 512, kernel=4, stride=2, pad=1)` $\to$ `InstanceNorm` $\to$ `LeakyReLU(0.2)`
  - `down5`: `Conv2d(512, 512, kernel=4, stride=2, pad=1)` $\to$ `LeakyReLU(0.2)` (Bottleneck)
- **Decoder**:
  - 5 upsampling blocks using bilinear interpolation + $3\times 3$ Conv + `InstanceNorm` + `ReLU` + Skip Connections.
- **Outputs**:
  - `global_feature`: Latent tensor $[B, 128, 64, 64]$ passed to Local Enhancer.
  - `coarse_rgb`: Low-resolution output $[B, 3, 64, 64]$ via Tanh.

### 2.2. LocalEnhancer (Fine Network)
- **Input**: Full-resolution thermal tensor $[B, 2, 128, 128]$ and `global_feature` $[B, 128, 64, 64]$.
- **Encoder**:
  - `enc1`: `Conv2d(2, 32, kernel=4, stride=2, pad=1)` $\to$ `LeakyReLU(0.2)` $\to [B, 32, 64, 64]$
  - `enc2`: `Conv2d(32, 64, kernel=4, stride=2, pad=1)` $\to$ `InstanceNorm` $\to$ `LeakyReLU(0.2)` $\to [B, 64, 32, 32]$
- **Context Fusion**:
  - `fusion_proj`: `Conv2d(128, 64, kernel=1)` projects global feature channels.
  - Fusion operation: `fused = local_feature + fusion_proj(global_feature)`
- **Decoder**:
  - `dec1`: `Upsample(2x, bilinear)` $\to$ `Conv2d(64, 32, kernel=3, pad=1)` $\to$ `InstanceNorm` $\to$ `ReLU`
  - `final_up`: `Upsample(2x, bilinear)` $\to$ `Conv2d(32, 3, kernel=3, pad=1)` $\to$ `Tanh` $\to [B, 3, 128, 128]$

---

## 3. Discriminator Specification (`MultiScaleDiscriminator`)

**File**: [`src/models/pix2pix/discriminator.py`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/models/pix2pix/discriminator.py)  
**Total Parameters**: **5,532,418** (5.53M across 2 scales)  
**Input Shape**: `[B, 5, 128, 128]` (Concatenation of 2 IR input channels + 3 RGB channels)  

### 3.1. Architecture & Lipschitz Continuity
- Operates across 2 scales:
  - **Scale 0**: Full resolution $[B, 5, 128, 128]$.
  - **Scale 1**: $2\times$ downsampled $[B, 5, 64, 64]$ via `AvgPool2d(kernel_size=3, stride=2, pad=1)`.
- Each scale uses an independent `PatchDiscriminator` with a $70\times 70$ receptive field:
  - `initial`: `SpectralNorm(Conv2d(5, 64, kernel=4, stride=2, pad=1))` $\to$ `LeakyReLU(0.2)`
  - `block1`: `SpectralNorm(Conv2d(64, 128, kernel=4, stride=2, pad=1))` $\to$ `LeakyReLU(0.2)`
  - `block2`: `SpectralNorm(Conv2d(128, 256, kernel=4, stride=2, pad=1))` $\to$ `LeakyReLU(0.2)`
  - `block3`: `SpectralNorm(Conv2d(256, 512, kernel=4, stride=1, pad=1))` $\to$ `LeakyReLU(0.2)`
  - `final`: `SpectralNorm(Conv2d(512, 1, kernel=4, stride=1, pad=1))` $\to$ Logit Map $[B, 1, 30, 30]$
- **No InstanceNorm**: Instance normalization rescales activations dynamically per sample, which undoes the 1-Lipschitz condition enforced by `SpectralNorm`. Spectral normalization alone enforces strict stability across multi-scale branches.