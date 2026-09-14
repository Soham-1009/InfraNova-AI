# Training Pipeline Documentation — InfraNova AI

**Document Version:** 2.1 (Exp9 Production Specification)  
**Date:** 2026-09-13  
**Module**: `src/training/` & `kaggle_kernel/`  

---

## 1. Multi-Term Loss Objective

The GAN objective is minimized via two alternating Adam optimizers:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + \lambda_{L1}\mathcal{L}_{L1} + \lambda_{\text{perc}}\mathcal{L}_{\text{perc}} + \lambda_{\text{ssim}}\mathcal{L}_{\text{ssim}} + \lambda_{\text{chroma}}\mathcal{L}_{\text{chroma}} + \lambda_{\text{feat}}\mathcal{L}_{\text{feat}} + \lambda_{\text{sat}}\mathcal{L}_{\text{sat}}$$

### Hyperparameter Configurations:

| Hyperparameter | Legacy Baseline (`configs/config.yaml`) | Active Production Exp9 (`configs/exp9_resnet_generator.yaml`) |
| :--- | :---: | :---: |
| **Generator Architecture** | `Pix2PixHDGenerator` (21.38M) | `Pix2PixHDGlobalResNetGenerator` (11.37M) |
| **Adversarial Objective** | BCE (`gan_mode: bce`) | LSGAN (`gan_mode: lsgan`, MSE) |
| $\lambda_{\text{adv}}$ | 1.0 | 1.0 |
| $\lambda_{L1}$ | 10.0 | 10.0 |
| $\lambda_{\text{perc}}$ (VGG-19) | 10.0 | **5.0** (Prevents over-smoothing) |
| $\lambda_{\text{ssim}}$ | 5.0 | 5.0 |
| $\lambda_{\text{chroma}}$ | 2.0 | 2.0 |
| $\lambda_{\text{feat}}$ | 5.0 | **0.0** (Ablated in Exp2; omitted in Exp9) |
| $\lambda_{\text{sat}}$ (CIE Lab Saturation) | 0.0 | **0.05** ($\varepsilon = 10^{-6}$ for stable gradients) |

---

## 2. Optimizer & Learning Rate Schedule

- **Optimizers**: Adam for both Generator and Discriminators ($\beta_1 = 0.5, \beta_2 = 0.999$, initial $\text{lr} = 2\times 10^{-4}$).
- **Linear Learning Rate Annealing**:
  - Epochs 1–50 (or 1–100): Constant $\text{lr} = 2\times 10^{-4}$.
  - Subsequent epochs: Linear decay down to $\eta_{\text{min}} = 1\times 10^{-6}$:
    $$\text{lr}(e) = \eta_{\text{min}} + (\text{lr}_{\text{base}} - \eta_{\text{min}})\left(1 - \frac{e - e_{\text{decay}}}{e_{\text{total}} - e_{\text{decay}}}\right)$$
- **Early Stopping**: Early stopping monitor watches validation SSIM with patience (`patience: 25`). In Exp9, peak SSIM occurred at epoch 46, and early stopping cleanly terminated training at epoch 71.
- **Long-Training Study**: Expanding training budget to 250 epochs (patience 100) on Tesla T4 resulted in 84 non-improving epochs post-epoch 46, causing spectral overfitting and saturation inflation. Further epoch increases are exhausted; Exp9 epoch 46 is the finalized production model.

---

## 3. GPU Memory Stability & Checkpoint Management

For long-running training runs:
1. **CPU State Dict Serialization**: When saving checkpoints, model parameters are cloned to CPU memory (`{k: v.detach().cpu() for k, v in model.state_dict().items()}`), preventing GPU VRAM fragmentation.
2. **Rolling Checkpoints**: Maintains rolling checkpoints for `latest.pth`, `best_checkpoint.pth`, and metric-specific peaks (`best_ssim.pth`, `best_psnr.pth`, `best_lab.pth`, `best_sam.pth`, `best_sat_ratio.pth`).
3. **Explicit Garbage Collection**: Invokes `torch.cuda.empty_cache()` and `gc.collect()` at epoch boundaries.
4. **NaN/Inf Shielding**: Gradient clipping and `Trainer._is_finite()` validation protect both generator and discriminator from non-finite weight poisoning during training.
