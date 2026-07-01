## 1. Training Timeline (250 Epochs)

Training a GAN is not like training a standard classifier; the loss curves will not smoothly drop to zero. Here is the expected visual progression:

- **Epoch 1-5 (Initial Behaviour):** Model prioritises L1 loss. Outputs look like blurry, sepia-toned, or greyscale approximations. Edges are soft.
- **Epoch 10-30 (First Colours):** Discriminator forces real colours. "Blobs" of green for vegetation and grey for roads appear, but boundaries bleed.
- **Epoch 30-60 (Structural Improvement):** Model learns IR-to-geometry mapping. Building footprints and road networks become sharply defined.
- **Epoch 60-100 (Fine Textures):** High-frequency details emerge. Canopy texture of trees, lane markings, vehicle outlines visible.
- **Epoch 100-230 (Final Refinement):** Colour bleeding stops. Model achieves perceptual realism while the learning rate is still constant.
- **Epoch 230-250 (LR Decay):** Linearly decay the learning rate to stabilize the final checkpoint. Stop earlier if validation SSIM plateaus for the configured patience window.

---

## 2. Monitoring Dashboard

**Use Weights & Biases (W&B) over TensorBoard.** W&B tracks metrics in the cloud natively - if Colab disconnects, logs are safe and shareable for pitch deck.

### Real-Time Metrics to Log

| Metric | Healthy Range |
|--------|---------------|
| Loss_G_cGAN | 1.0 - 2.0 oscillation |
| Loss_G_L1 | Steady decrease, plateaus |
| Loss_D | 0.5 - 0.7 (NEVER 0.0) |
| Val_PSNR | Track every 5 epochs |
| Val_SSIM | Track every 5 epochs |

### Visual Strategy

Log a fixed grid of 4 test images to W&B at the end of EVERY epoch. Watching the same 4 images evolve is the best way to spot colour washing.

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

Colab instances wipe data on disconnect. MUST save to Google Drive.

- **Path:** `/content/drive/MyDrive/InfraNova-AI/checkpoints/`
- **Naming:** `pix2pix_epoch_{epoch}_PSNR_{psnr:.2f}.pth`
- **Keep Top 3:** Save only when val SSIM/PSNR improves, keep best 3 files
- **Resume Logic:** Save optimizer states alongside weights for clean resumption

---

## 5. Validation Strategy

- **Frequency:** Every 5 epochs (saves compute)
- **Qualitative Set:** Hardcode 8 diverse IR images (2 urban, 2 forest, 2 water, 2 cloudy)
- **Decision Matrix:**
  - PSNR > 24 and SSIM > 0.75 by Epoch 100: Freeze model, move to UI dev
  - Metrics plateau at Epoch 80: Trigger LR decay

---

## 6. Resource Planning (Colab Limits)

- **12-Hour Limit:** Free Colab has hard 12-hour limit, ~90 min idle timeout
- **If browser sleeps:** Training dies
- **Recommended:** Colab Pro (~$10) for Background Execution during hackathon week
- **Disconnection Handling:** Wrap training in `try/except KeyboardInterrupt` to save state cleanly

---

## 7. Pitch Deck Material (Capture During Training)

Don't wait until Day 13. Capture during Day 5-7 training:

1. **The Evolution GIF:** W&B logged images from Epoch 1, 10, 50, 150, 250 stitched into 4-second GIF
2. **The "Ah-Ha" Graph:** Screenshot of D and G equilibrium point (~Epoch 40)
3. **Before/After Grids:** Raw IR vs generated RGB, highlight cases where human interpretation fails (dark road vs dark river)

---

## Key Decisions Summary

| Decision | Choice |
|----------|--------|
| Logging tool | Weights & Biases (not TensorBoard) |
| Validation frequency | Every 5 epochs |
| Sample logging | Every epoch (4 fixed images) |
| Checkpoint policy | Top 3 by val SSIM + best |
| Storage | Google Drive (not local Colab) |
| Resource tier | Colab Pro recommended |
"""
