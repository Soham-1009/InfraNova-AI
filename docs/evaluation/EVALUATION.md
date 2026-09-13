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
### 2.3. Standardized Five-Way Evaluation Benchmark (1,259 Held-Out Test Samples)

To evaluate architectural and loss advancements across iterations, all major candidates were benchmarked under identical standardized conditions on the 1,259-sample test split:

| Metric | Production *(Frozen)* | Exp6 | Exp7 | Long Exp7 | Exp9 ResNet *(Audited Candidate)* | Exp9 Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Generator Architecture** | Custom U-Net + Local | Custom U-Net + Local | Custom U-Net + Local | Custom U-Net + Local | **Global ResNet (9 Blocks)** | — |
| **Generator Parameters** | 21,383,238 | 21,383,238 | 21,383,238 | 21,383,238 | **11,369,795** | **-46.83% (-10.01M)** |
| **PSNR (dB)** $\uparrow$ | 11.714 dB | 12.786 dB | 12.948 dB | 12.926 dB | **13.745 dB** | **+2.031 dB** |
| **SSIM** $\uparrow$ | 0.3254 | 0.3054 | 0.3157 | 0.3178 | **0.4502** | **+38.3% (+0.1248)** |
| **MAE** $\downarrow$ | 0.2144 | 0.1993 | 0.1950 | 0.1918 | **0.1723** | **-19.6%** |
| **RMSE** $\downarrow$ | 0.2752 | 0.2467 | 0.2436 | 0.2411 | **0.2217** | **-19.5%** |
| **SAM (rad / deg)** $\downarrow$ | 0.2187 ($12.53^\circ$) | 0.1962 ($11.24^\circ$) | 0.2012 ($11.53^\circ$) | 0.2013 ($11.53^\circ$) | **0.1968 ($11.28^\circ$)** | **-10.0% (-1.25°)** |
| **CIE Lab $\Delta E$** $\downarrow$ | 27.235 | 24.686 | 24.437 | 24.243 | **22.571** | **-17.1%** |
| **Saturation Ratio** | 1.1993 | 0.5837 | 0.9310 | 0.8706 | **0.9340** | **Natural Fidelity** |
| **Saturation Deviation** $\downarrow$ | 0.1993 | 0.4163 | 0.0690 | 0.1294 | **0.0660** | **-66.9% Sat Error** |
| **Histogram Distance** $\downarrow$ | **0.3572** | 0.5479 | 0.5862 | 0.5886 | 0.4607 | Trade-off (+0.1035) |

---

## 3. Standardized Downstream YOLOv8 Detection Evaluation

Downstream feature and infrastructure utility is benchmarked using an off-the-shelf YOLOv8n detector across all 1,259 test samples (`conf=0.15`, `imgsz=128`, greedy matching on ground truth bounding boxes):

| Model Candidate | Total Detections | Precision @ 0.25 | Recall @ 0.25 | F1 @ 0.25 | Precision @ 0.50 | Recall @ 0.50 | F1 @ 0.50 | Mean Matched IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production Baseline** | 91 (38 TP, 53 FP) | 0.4176 | 0.2331 | 0.2992 | 0.3956 | 0.2209 | 0.2835 | 0.9977 |
| **Exp6 (LSGAN)** | 88 (39 TP, 49 FP) | 0.4432 | 0.2393 | 0.3108 | 0.4318 | 0.2331 | 0.3028 | 0.9979 |
| **Exp7 (LSGAN + Sat 0.05)** | 61 (33 TP, 28 FP) | **0.5410** | 0.2025 | 0.2946 | **0.5246** | 0.1963 | 0.2857 | 0.9982 |
| **Long Exp7 (200 Epochs)** | 74 (36 TP, 38 FP) | 0.4865 | 0.2209 | 0.3038 | 0.4730 | 0.2147 | 0.2954 | **0.9983** |
| **Exp9 (Global ResNet)** | **128 (54 TP, 74 FP)** | 0.4219 | **0.3313** | **0.3711** | 0.4141 | **0.3252** | **0.3643** | 0.9980 |

### Key Findings
1. **Recall Surge**: Exp9 dramatically increases detection recall (+42.1% relative gain: 54 true positives vs 38 in Production and 33 in Exp7), driven by sharper, hallucination-free boundary reconstruction.
2. **Net F1 Superiority**: Despite generating more candidates, Exp9 delivers the highest overall F1 score across all experiments (**0.3711 @ 0.25**, **0.3643 @ 0.50**), a +24.0% to +28.5% gain over Production.
3. **Geometric Precision**: All matched detections retain a mean IoU of $\approx 0.9980$, confirming precise spatial co-registration.
