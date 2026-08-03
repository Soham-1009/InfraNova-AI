## 1. Training Timeline (250 Epochs)

Training a GAN is not like training a standard classifier; the loss curves will not smoothly drop to zero. Here is the expected visual progression:

- **Epoch 1–5 (Initial Behaviour):** Model prioritises L1 loss. Outputs look like blurry, sepia-toned, or greyscale approximations. Edges are soft.
- **Epoch 10–30 (First Colours):** Discriminator forces real colours. "Blobs" of green for vegetation and grey for roads appear, but boundaries bleed.
- **Epoch 30–60 (Structural Improvement):** Model learns IR-to-geometry mapping. Building footprints and road networks become sharply defined.
- **Epoch 60–100 (Fine Textures):** High-frequency details emerge. Canopy texture of trees, lane markings, vehicle outlines visible.
- **Epoch 100–230 (Final Refinement):** Colour bleeding stops. Model achieves perceptual realism while the learning rate is still constant.
- **Epoch 230–250 (LR Decay):** Linearly decay the learning rate to stabilize the final checkpoint. Stop earlier if validation SSIM plateaus for the configured patience window.

---

## 2. Monitoring

Training metrics are logged to CSV (`logs/training.csv`) and optionally to TensorBoard (`logs/tensorboard/`). Logging is configured in `configs/config.yaml`.

### Real-Time Metrics to Log

| Metric | Healthy Range |
|--------|---------------|
| Loss_G_cGAN | 1.0 - 2.0 oscillation |
| Loss_G_L1 | Steady decrease, plateaus |
| Loss_D | 0.5 - 0.7 (NEVER 0.0) |
| Val_PSNR | Track every 5 epochs |
| Val_SSIM | Track every 5 epochs |

### Visual Strategy

Log a fixed grid of 4 test images to `outputs/visualizations/` at the end of every validation epoch. Watching the same 4 images evolve is the best way to spot colour washing.

---

## 3. Failure Mode Playbook

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| D loss to 0.0, G loss explodes | Discriminator Overpowering | Label smoothing (real=0.9), reduce D LR, update D once per 2 G updates |
| Same output regardless of input | Mode Collapse | Increase batch size, gradient penalty, verify shuffling |
| Dull/greyish outputs | Colour Washing (L1 dominating) | Decrease L1 weight from 100 to 50 |
| Grid-like patterns | Checkerboard Artefacts | Verify ConvTranspose2d kernel=4 stride=2 |
| Val loss up, train loss down | Overfitting (2000 pairs is small) | Aggressive augmentation |

---

## 4. Checkpoint Management

Checkpoints are saved to `outputs/models/`.

- **Best checkpoint:** `outputs/models/best/pix2pix_landsat_best.pth`
- **Latest checkpoint:** `outputs/models/final/pix2pix_landsat_latest.pth`
- **Final checkpoint:** `outputs/models/final/pix2pix_landsat_final.pth`
- **Resume Logic:** Full optimizer and scheduler state is saved alongside model weights for deterministic resumption.

On Kaggle, checkpoints persist in `/kaggle/working/outputs/models/` and are available as notebook output after the session ends.

---

## 5. Validation Strategy

- **Frequency:** Every 5 epochs (`training.sample_every: 5` in config)
- **Qualitative Set:** Hardcode 8 diverse IR images (2 urban, 2 forest, 2 water, 2 cloudy)
- **Decision Matrix:**
  - PSNR > 24 and SSIM > 0.75 by Epoch 100: Freeze model, move to evaluation
  - Metrics plateau at Epoch 80: Trigger LR decay

---

## 6. Resource Planning

### Kaggle (Primary)
- **Session limit:** 12 hours GPU, 9 hours for P100/T4
- **Checkpointing:** Every 10 epochs to survive session timeouts
- **Resume:** Set `training.resume_from` in config to resume from the latest checkpoint

### Local (CPU/GPU)
- No session limits; use `venv` environment
- Monitor GPU memory with `nvidia-smi`
- For CPU-only training, disable AMP (`training.amp: false`)

---

## 7. Capture During Training

Capture these materials during training for documentation and evaluation:

1. **The Evolution GIF:** Logged images from Epoch 1, 10, 50, 150, 250 stitched into a 4-second GIF
2. **The Equilibrium Graph:** Screenshot of D and G loss equilibrium point (~Epoch 40)
3. **Before/After Grids:** Raw IR vs generated RGB, highlight cases where interpretation is difficult (dark road vs dark river)

---

## Key Decisions Summary

| Decision | Choice |
|----------|--------|
| Logging tool | CSV + TensorBoard (local) |
| Validation frequency | Every 5 epochs |
| Sample logging | Every validation epoch (4 fixed images) |
| Checkpoint policy | Best by val SSIM + latest + final |
| Storage | `outputs/models/` (local) or `/kaggle/working/outputs/models/` (Kaggle) |
| Training platform | Kaggle GPU (primary), local GPU/CPU (secondary) |
