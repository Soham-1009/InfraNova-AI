# Model Architecture Specification — InfraNova AI

**Document Version:** 2.1 (Exp9 Production Specification)  
**Date:** 2026-09-13  
**Status:** Authoritative Production Model Specification  

---

## 1. Architectural Overview

InfraNova AI employs a **conditional Generative Adversarial Network** tailored specifically for dual-band thermal radiance input ($128\times 128$) to true color optical RGB synthesis.

The active production serving model is **Exp9**, powered by a compact **Global ResNet Generator** (`Pix2PixHDGlobalResNetGenerator`, 11.37M parameters) paired with a **MultiScaleDiscriminator** (5.53M parameters). The previous dual-branch U-Net generator (`Pix2PixHDGenerator`, 21.38M parameters) is preserved in the codebase as an audited legacy baseline.

```
+-----------------------------------------------------------------------------+
|                                 INPUT TENSOR                                |
|             [B, 2, 128, 128] (Band 10 + Band 11, Range: [-1, 1])            |
+-----------------------------------------------------------------------------+
                                       |
                   ▼ (7×7 Conv, ReflectionPad, 64ch, InstanceNorm)
                                [Initial Stage]
                                       │
                   ▼ (Stride 2 Conv, 128ch, 64×64)
                              [Downsampling Stage 1]
                                       │
                   ▼ (Stride 2 Conv, 256ch, 32×32)
                              [Downsampling Stage 2]
                                       │
                   ▼ (9 ResNet Blocks, 256ch, 32×32)
                             [ResNet Bottleneck 9×]
                                       │
                   ▼ (Stride 2 ConvTranspose, 128ch, 64×64)
                               [Upsampling Stage 1]
                                       │
                   ▼ (Stride 2 ConvTranspose, 64ch, 128×128)
                               [Upsampling Stage 2]
                                       │
                   ▼ (7×7 Conv, ReflectionPad, 3ch) + Tanh
                              [Output Stage [-1, 1]]
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
|                             GENERATED RGB IMAGE                             |
|                        [B, 3, 128, 128], Tanh [-1, 1]                       |
+-----------------------------------------------------------------------------+
```

---

## 2. Active Production Generator (`Pix2PixHDGlobalResNetGenerator` — Exp9)

**File**: [`src/models/pix2pix/generator_hd.py`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/models/pix2pix/generator_hd.py)  
**Total Parameters**: **11,369,795** (11.37M — 46.83% parameter reduction vs. legacy baseline)  
**Input Shape**: `[B, 2, 128, 128]`  
**Output Shape**: `[B, 3, 128, 128]` (Range $[-1, 1]$ via Tanh)  

### 2.1. Architectural Design & Inductive Bias
Inspired by the Johnson et al. and Pix2PixHD Global Generator design, this network trades multi-scale U-Net branches for a streamlined, deep residual pipeline operating at constant feature resolution:

1. **Initial Processing Layer**:
   - `ReflectionPad2d(3)` $\to$ `Conv2d(2, 64, kernel=7, padding=0, bias=False)` $\to$ `InstanceNorm2d(64)` $\to$ `ReLU(inplace=True)`
   - Spatial dimension: $[B, 64, 128, 128]$.
2. **Strided Downsampling (2 Stages)**:
   - Down 1: `Conv2d(64, 128, kernel=3, stride=2, padding=1, bias=False)` $\to$ `InstanceNorm2d(128)` $\to$ `ReLU` $\to [B, 128, 64, 64]$
   - Down 2: `Conv2d(128, 256, kernel=3, stride=2, padding=1, bias=False)` $\to$ `InstanceNorm2d(256)` $\to$ `ReLU` $\to [B, 256, 32, 32]$
3. **Deep Residual Engine (9 ResnetBlocks)**:
   - Exactly **9 residual blocks** operating at constant feature dimension $[B, 256, 32, 32]$.
   - Each `ResnetBlock`:
     $$\mathbf{x} + \text{InstanceNorm}\left(\text{Conv}_{3\times 3}\left(\text{Pad}\left(\text{ReLU}\left(\text{InstanceNorm}\left(\text{Conv}_{3\times 3}\left(\text{Pad}(\mathbf{x})\right)\right)\right)\right)\right)\right)$$
   - Uses reflection padding (`ReflectionPad2d(1)`) to eliminate boundary gradient distortions.
4. **Transposed Convolution Upsampling (2 Stages)**:
   - Up 1: `ConvTranspose2d(256, 128, kernel=3, stride=2, padding=1, output_padding=1, bias=False)` $\to$ `InstanceNorm2d(128)` $\to$ `ReLU` $\to [B, 128, 64, 64]$
   - Up 2: `ConvTranspose2d(128, 64, kernel=3, stride=2, padding=1, output_padding=1, bias=False)` $\to$ `InstanceNorm2d(64)` $\to$ `ReLU` $\to [B, 64, 128, 128]$
5. **Final Output Layer**:
   - `ReflectionPad2d(3)` $\to$ `Conv2d(64, 3, kernel=7, padding=0)` $\to$ `Tanh()` $\to [B, 3, 128, 128]$.

### 2.2. Performance & Efficiency Advantages
- **No Skip-Connection Contamination**: Thermal features are translated into semantic representations rather than directly copied across long skip connections, eliminating high-frequency noise and magenta chromatic artifacts.
- **Superior Generalization**: Achieves **0.4502 SSIM** (+38.3% over legacy Production) and **13.745 dB PSNR** (+2.03 dB over legacy Production) on 1,259 test samples.

---

## 3. Legacy Generator Baseline (`Pix2PixHDGenerator`)

**File**: [`src/models/pix2pix/generator_hd.py`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/models/pix2pix/generator_hd.py)  
**Total Parameters**: **21,382,915** (21.38M)  
**Input Shape**: `[B, 2, 128, 128]`  
**Output Shape**: `[B, 3, 128, 128]` (Range $[-1, 1]$ via Tanh)  

### 3.1. GlobalGenerator (Coarse Network)
- **Input**: Downsampled thermal tensor $[B, 2, 64, 64]$ via `F.avg_pool2d(kernel_size=2, stride=2)`.
- **Encoder**: 6 downsampling conv blocks reaching 512 channels.
- **Decoder**: 5 upsampling blocks using bilinear interpolation + $3\times 3$ Conv + `InstanceNorm` + `ReLU` + Skip Connections.
- **Outputs**: Latent tensor $[B, 128, 64, 64]$ and coarse RGB estimate $[B, 3, 64, 64]$.

### 3.2. LocalEnhancer (Fine Network)
- **Input**: Full-resolution thermal tensor $[B, 2, 128, 128]$ and `global_feature` $[B, 128, 64, 64]$.
- **Encoder**: 2 strided conv blocks reaching 64 channels at $32\times 32$.
- **Context Fusion**: Projects global feature to 64 channels via $1\times 1$ conv and adds to local feature.
- **Decoder**: Bilinear upsampling to $128\times 128$ with $3\times 3$ Conv and Tanh.

---

## 4. Discriminator Specification (`MultiScaleDiscriminator`)

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