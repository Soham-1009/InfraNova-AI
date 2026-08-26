# Experimental History & Telemetry — InfraNova AI

**Document Version:** 2.0 (Complete 250-Epoch Experimental Report)  
**Date:** 2026-08-26  
**Auditor:** ML Engineer & Experimental Scientist  

---

## 1. Experimental Overview & Progression

The InfraNova AI cross-spectral translation model was trained across a systematic 3-stage progression totaling **250 epochs** and **8.21 GPU hours** on dual NVIDIA Tesla T4 GPUs (`DataParallel`, batch size 64):

1. **Stage 1: Initial Ascent (Epochs 1–100)**: Constant learning rate $\text{lr} = 2\times 10^{-4}$. Rapid structural and perceptual feature acquisition.
2. **Stage 2: Linear Annealing (Epochs 101–200)**: Learning rate decayed linearly from $2\times 10^{-4}$ to $6.7\times 10^{-5}$. Significant stabilization of color saturation and reduction of adversarial high-frequency noise.
3. **Stage 3: Fine Convergence & Terminal Plateau (Epochs 201–250)**: Fine weight refinement down to $\eta_{\text{min}} = 1\times 10^{-6}$. Saturation rolling variance dropped to an all-time low of $\sigma = 0.0386$.

---

## 2. Quantitative Performance Across Training Stages

*Source: `outputs/final/training_master_250epochs.csv` and `master_summary_statistics.json`.*

| Metric | Stage 1 Mean (Ep 1–100) | Stage 2 Mean (Ep 101–200) | Stage 3 Mean (Ep 201–250) | Final Model (Epoch 250) | All-Time Best Record |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** | 11.139 dB | 11.333 dB | **11.499 dB** | 11.494 dB | **11.754 dB** *(Ep 137)* |
| **SSIM Index** | 0.1900 | 0.2324 | **0.2333** | 0.2340 | **0.2522** *(Ep 68)* |
| **MAE** | 0.2388 | 0.2265 | **0.2220** | 0.2217 | **0.2183** *(Ep 67)* |
| **RMSE** | 0.2980 | 0.2865 | **0.2815** | 0.2810 | **0.2741** *(Ep 67)* |
| **CIE $\Delta E^*_{ab}$** | 30.35 | 30.52 | **29.81** | 29.87 | **27.61** *(Ep 40)* |
| **SAM (rad)** | 0.2450 | 0.2280 | **0.2245** | 0.2257 | **0.2050** *(Ep 127)* |
| **Saturation Ratio** | 0.9364 ($\sigma=0.300$) | 0.9553 ($\sigma=0.307$) | **0.8948** ($\sigma=\mathbf{0.157}$) | 0.9035 | **1.0033** *(Ep 92)* |
| **10-Ep Saturation $\sigma$** | 0.3632 | 0.1245 | **0.0386** | 0.0386 | **0.0386** *(Ep 241–250)* |
| **Histogram Dist** | 0.1420 | 0.1180 | **0.0950** | 0.0929 | **0.0843** *(Ep 161)* |

---

## 3. Checkpoint Inventory

| Checkpoint | Path | File Size | Verified Epoch | Role |
| :--- | :--- | :---: | :---: | :--- |
| **`epoch_250.pth`** | `outputs/final/epoch_250.pth` | 323.2 MB | Epoch 250 | Final converged weights at completion |
| **`best_checkpoint.pth`** | `outputs/final/best_checkpoint.pth` | 323.2 MB | Multi-Criteria | Global optimal model |
| **`best_ssim.pth`** | `outputs/final/best_ssim.pth` | 323.2 MB | Epoch 68 | Peak structural similarity (0.2522) |
| **`best_psnr.pth`** | `outputs/final/best_psnr.pth` | 323.2 MB | Epoch 137 | Peak signal-to-noise ratio (11.75 dB) |
| **`best_lab.pth`** | `kaggle_kernel/.../best_lab.pth` | 323.2 MB | Epoch 40 | Minimal perceptual color error (27.61) |
| **`best_sam.pth`** | `kaggle_kernel/.../best_sam.pth` | 323.2 MB | Epoch 127 | Lowest spectral angle distortion (0.2050) |
| **`best_sat_ratio.pth`** | `kaggle_kernel/.../best_sat_ratio.pth` | 323.2 MB | Epoch 92 | Optimal saturation ratio (1.0033) |
