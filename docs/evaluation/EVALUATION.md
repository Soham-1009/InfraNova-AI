# Evaluation & Metrics Documentation — InfraNova AI

**Document Version:** 2.0  
**Date:** 2026-08-26  
**Module**: `scripts/evaluation/` & `src/training/losses.py`  

---

## 1. Mathematical Metric Formulations

### 1.1. Structural Similarity Index (SSIM)
Evaluates luminance, contrast, and structural preservation:
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
*Where $C_1 = 0.01^2, C_2 = 0.03^2$. Higher is better ($1.0$ = identical).*

### 1.2. Peak Signal-to-Noise Ratio (PSNR)
Measures pixel-level reconstruction fidelity in decibels:
$$\text{PSNR} = 10 \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$$
*Where $\text{MAX}_I = 1.0$ on normalized $[0, 1]$ images. Higher is better.*

### 1.3. Spectral Angle Mapper (SAM)
Computes the mean spectral vector angle across pixels in radians:
$$\text{SAM}(x, y) = \frac{1}{N}\sum_{i=1}^N \arccos\left(\frac{x_i \cdot y_i}{\|x_i\|_2 \|y_i\|_2}\right)$$
*Lower is better ($0.0$ = identical spectral angle).*

### 1.4. CIE $\Delta E^*_{ab}$ Perceptual Color Difference
Approximates perceptual human color sensitivity in $L^*a^*b^*$ space:
$$\Delta E^*_{ab} = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}$$
*Lower is better.*

### 1.5. Optical Saturation Ratio
Measures whether synthesized images suffer from desaturation collapse:
$$\text{Ratio}_{\text{sat}} = \frac{\mathbb{E}[\sigma_{\text{RGB}}(\hat{y})]}{\mathbb{E}[\sigma_{\text{RGB}}(y)]}$$
*Ideal target is $1.0$. Values $<1.0$ indicate desaturation; values $>1.0$ indicate oversaturation.*

### 1.6. Color Histogram Wasserstein Distance
Evaluates global RGB tonal distribution alignment via 64-bin chi-squared distance across color channels.

---

## 2. Model Selection & Generalization Testing Protocol

### 2.1. Validation-Only Selection Heuristic (Leakage-Free Protocol)
To ensure strict scientific validity and prevent test-set contamination:
1. Candidate checkpoints from the 250-epoch dual-band run were evaluated on the **1,432-sample validation split** (`data/landsat9_b10_b11/splits/val/`).
2. Candidates were ranked using a multi-metric composite optimization heuristic defined *a priori*:
   $$\text{Score}_{\text{val}} = (10.0 \cdot \text{SSIM}_{\text{val}}) + (0.5 \cdot \text{PSNR}_{\text{val}}) - (0.1 \cdot \text{CIE }\Delta E^*_{ab}) - (2.0 \cdot \text{SAM}_{\text{val}})$$
   *Subject to the optical saturation constraint: $|\text{SatRatio} - 1.0| \le 0.20$.*
3. **`best_ssim.pth`** (saved at **Epoch 223**, SHA256: `4604d36d07a0fb4c...8baf6aa8`) achieved the top composite score (**5.741**, Val SSIM: 0.3257 / 0.2410 windowed, Val PSNR: 11.599 dB, Sat Error: 0.1048) and was frozen as the official application model [`outputs/best/pix2pix_landsat_best.pth`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/outputs/best/pix2pix_landsat_best.pth).

### 2.2. Official Unbiased Test Scorecard (1,259 Held-Out Samples)
Evaluated once on the unseen, held-out test split (`data/landsat9_b10_b11/splits/test/`):

| Metric | Single-Band Baseline (Ep 100) | **Frozen Application Model (`best_ssim.pth`, Ep 223)** | Generalization Delta |
| :--- | :---: | :---: | :---: |
| **Test PSNR** $\uparrow$ | 10.820 dB | **11.714 dB** ($\pm 3.229\text{ dB}$) | **+0.894 dB** |
| **Test SSIM (11×11 Gaussian Windowed)** $\uparrow$ | 0.1841 | **0.2347** ($\pm 0.1630$) | **+27.5%** |
| **Test SSIM (Global Image-Level)** $\uparrow$ | — | **0.3254** ($\pm 0.3354$) | — |
| **Test MAE** $\downarrow$ | 0.2512 | **0.2144** ($\pm 0.0774$) | **-14.6%** |
| **Test RMSE** $\downarrow$ | 0.3120 | **0.2752** ($\pm 0.0842$) | **-11.8%** |
| **CIE $\Delta E^*_{ab}$ Error** $\downarrow$ | 35.420 | **27.235** ($\pm 8.294$) | **-23.1%** |
| **Spectral Angle (SAM)** $\downarrow$ | 0.2680 rad | **0.2187 rad ($12.53^\circ$)** | **-18.4%** |
| **Saturation Ratio** (Target: 1.0) | 0.6023 (Err: 0.398) | **1.1993 (Err: 0.199)** | **-49.9% Error** |
| **Color Histogram Distance** $\downarrow$ | 0.4840 | **0.3572** ($\pm 0.322$) | **-26.2%** |

> **SSIM Mathematical Resolution**: The training telemetry used localized **$11\times 11$ Gaussian-windowed SSIM ($\sigma=1.5$)** evaluating local patch correlations, giving **0.2410** on validation and **0.2347** on test. When computed via whole-image global spatial pooling (Global SSIM), whole-image averages smooth high-frequency texture variation, producing **0.3257** and **0.3254**.

---

## 3. Downstream Detection Evaluation Methodology

Downstream object and feature extraction was evaluated using paired bounding box matching against Ground Truth Optical RGB (IoU threshold $\ge 0.25$):
- **Conclusion**: The generated imagery did not improve downstream object detection performance under this evaluation protocol. Both raw thermal and synthesized RGB retained very low precision ($0.016$) and recall ($0.029$) on generic COCO objects.
- **False-Positive Suppression**: The validation-selected checkpoint (`best_ssim.pth`, Epoch 223) suppressed spurious detector triggers induced by late-stage adversarial noise from $112 \to 64$ activations compared to `epoch_250.pth`.

---

## 4. Running Standalone Model Evaluation

```powershell
# Run validation-only model selection and single test evaluation
python scripts/evaluation/validation_model_selection_and_test.py
```
