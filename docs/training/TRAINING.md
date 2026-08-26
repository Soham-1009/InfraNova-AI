# Training Pipeline Documentation — InfraNova AI

**Document Version:** 2.0  
**Date:** 2026-08-26  
**Module**: `src/training/` & `kaggle_kernel/`  

---

## 1. Multi-Term Loss Objective

The GAN objective is minimized via two alternating Adam optimizers:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + \lambda_{L1}\mathcal{L}_{L1} + \lambda_{\text{perc}}\mathcal{L}_{\text{perc}} + \lambda_{\text{ssim}}\mathcal{L}_{\text{ssim}} + \lambda_{\text{chroma}}\mathcal{L}_{\text{chroma}} + \lambda_{\text{feat}}\mathcal{L}_{\text{feat}}$$

### Configured Hyperparameters (`configs/config.yaml`):
- $\lambda_{\text{adv}} = 1.0$ (Adversarial loss on multi-scale discriminator outputs)
- $\lambda_{L1} = 10.0$ (Pixel L1 distance)
- $\lambda_{\text{perc}} = 10.0$ (VGG-19 perceptual distance across relu1_2, relu2_2, relu3_4, relu4_4)
- $\lambda_{\text{ssim}} = 5.0$ (Differentiable SSIM loss, $1 - \text{SSIM}$)
- $\lambda_{\text{chroma}} = 2.0$ (Color saturation std loss)
- $\lambda_{\text{feat}} = 5.0$ (Discriminator multi-scale feature matching)

---

## 2. Optimizer & Learning Rate Schedule

- **Optimizers**: Adam for both Generator and Discriminators ($\beta_1 = 0.5, \beta_2 = 0.999$, initial $\text{lr} = 2\times 10^{-4}$).
- **Linear Learning Rate Annealing**:
  - Epochs 1–100: Constant $\text{lr} = 2\times 10^{-4}$.
  - Epochs 101–250: Linear decay from $2\times 10^{-4}$ down to $\eta_{\text{min}} = 1\times 10^{-6}$ at Epoch 250:
    $$\text{lr}(e) = \eta_{\text{min}} + (\text{lr}_{\text{base}} - \eta_{\text{min}})\left(1 - \frac{e - e_{\text{decay}}}{e_{\text{total}} - e_{\text{decay}}}\right)$$

---

## 3. GPU Memory Stability & Checkpoint Management

For long-running training runs across 250 epochs:
1. **CPU State Dict Serialization**: When saving checkpoints, model parameters are cloned to CPU memory (`{k: v.detach().cpu() for k, v in model.state_dict().items()}`), preventing GPU VRAM fragmentation.
2. **Rolling Checkpoints**: Maintains rolling checkpoints for `latest.pth`, `best_checkpoint.pth`, and metric-specific peaks (`best_ssim.pth`, `best_psnr.pth`, `best_lab.pth`, `best_sam.pth`, `best_sat_ratio.pth`).
3. **Explicit Garbage Collection**: Invokes `torch.cuda.empty_cache()` and `gc.collect()` at epoch boundaries.
