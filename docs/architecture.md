# 🛰️ Architecture Review: Pix2Pix (cGAN) for IR-to-RGB Translation

**Project:** InfraNova AI (BAH 2026) | **Target Hardware:** Colab T4 (16GB VRAM)

### 📐 1. ARCHITECTURE DIAGRAMS (ASCII)

Pix2Pix relies on two competing networks: a **U-Net Generator** (to translate IR to RGB while preserving spatial resolution) and a **PatchGAN Discriminator** (to critique local image patches rather than the whole image at once).

#### A. U-Net Generator (Encoder-Decoder with Skip Connections)

**Input:** `256x256x1` (Grayscale IR) | **Output:** `256x256x3` (Colorized RGB)

```text
[INPUT IR] 256x256x1
      │
      ▼
(Conv down) ──> [E1] 128x128x64 ─────────────────────────────────────────┐ (Skip)
                     │                                                   │
                     ▼                                                   ▼
                [E2] 64x64x128 ─────────────────────────────────┐   [D1] 128x128x128 ──> (Conv up)
                     │                                          │        ▲
                     ▼                                          ▼        │
                [E3] 32x32x256 ─────────────────────────┐   [D2] 64x64x256
                     │                                  │        ▲
                     ▼                                  ▼        │
                [E4] 16x16x512 ─────────────────┐   [D3] 32x32x512
                     │                          │        ▲
                     ▼                          ▼        │
                [E5] 8x8x512 ───────────┐   [D4] 16x16x1024 (Concat 512+512)
                     │                  │        ▲
                     ▼                  ▼        │
                [E6] 4x4x512 ───┐   [D5] 8x8x1024
                     │          │        ▲
                     ▼          ▼        │
                [E7] 2x2x512    [D6] 4x4x1024
                     │               ▲
                     ▼               │
               [BOTTLENECK] 1x1x512 ─┘

                                                                       │
                                                                       ▼
                                                             [OUTPUT RGB] 256x256x3

```

*Note: The expanding path concatenates features from the contracting path (skip connections) to recover high-frequency spatial details lost during downsampling.*

#### B. 70x70 PatchGAN Discriminator

**Input:** `256x256x4` (IR and predicted/real RGB concatenated on channel axis)

```text
[IR + RGB Concat] 256x256x4
       │
       ▼ (Conv 4x4, Stride 2)
  128x128x64
       │
       ▼ (Conv 4x4, Stride 2)
   64x64x128
       │
       ▼ (Conv 4x4, Stride 2)
   32x32x256
       │
       ▼ (Conv 4x4, Stride 1)  <-- Note stride change
   31x31x512
       │
       ▼ (Conv 4x4, Stride 1)
[OUTPUT GRID] 30x30x1

```

*Note: Instead of a single "Real/Fake" scalar, PatchGAN outputs a 30x30 matrix. Each pixel in this matrix evaluates the authenticity of a 70x70 pixel "patch" in the original image. This forces the generator to perfect fine textures.*

---

### ⚙️ 2. TRAINING DYNAMICS EXPLANATION

Pix2Pix learns through a minimax game governed by a combined objective function. We use adversarial loss to get sharp colors, and L1 loss to enforce structural correctness:

$$G^* = \arg\min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

*(Typically, $\lambda = 100$ to ensure the L1 loss dominates the structural formation).*

#### Common Failure Modes & How to Spot Them:

* **Mode Collapse:** The generator finds one specific "green forest" texture that fools the discriminator and applies it everywhere, regardless of the input IR.
* *Symptom:* Discriminator loss drops to 0; Generator loss spikes. Output images look identically muddy.


* **Color Washing (Sepia/Desaturated look):**
* *Cause:* The L1 loss ($\lambda$) is too high relative to the cGAN loss. L1 penalizes distance, so the generator "plays it safe" by averaging colors into a dull, grayish-brown blur rather than risking sharp reds or greens.
* *Fix:* Decrease $\lambda$ slightly or increase discriminator learning rate.


* **Checkerboard Artifacts:**
* *Cause:* Transposed convolutions in the U-Net decoder have overlapping kernel/stride sizes (e.g., kernel size 3 with stride 2).
* *Fix:* Stick to kernel size 4 with stride 2, or replace `ConvTranspose2d` with `Bilinear Upsampling + standard Conv2d`.



---

### 📊 3. ARCHITECTURE COMPARISON TABLE

Why Pix2Pix is the optimal strategic choice for a 14-day hackathon on a Colab T4:

| Architecture | Paradigm | GPU Ram Needed | Train Time (2k dataset) | Inference Speed | Hackathon Suitability |
| --- | --- | --- | --- | --- | --- |
| **Pix2Pix** | Paired cGAN | ~4-6 GB | 6-8 Hours | **< 0.1s (Real-time)** | **Winner.** Fast to train, leaves time for UI. |
| **CycleGAN** | Unpaired GAN | ~8-12 GB | 18-24 Hours | < 0.2s | **Overkill.** We already have paired data. |
| **Diffusion (Palette)** | Generative Noise | 16GB+ (OOM risk) | 48+ Hours | ~5-10s (Iterative) | **Trap.** Too slow for a live web demo. |
| **Vision Transformers** | SwinIR/Restormer | ~12-16 GB | 30+ Hours | ~0.5s | **Risky.** Needs massive data to not overfit. |

---

### 📈 4. EXPECTED RESULTS PREVIEW (TIMELINE)

* **Epoch 1-5:** The model will act like an edge-detector. The output will look like a blurry, sepia-toned version of the IR image as L1 loss rapidly decreases.
* **Epoch 10-30:** Broad regional colors appear (green blobs for fields, gray blobs for roads). The PatchGAN discriminator forces the generator to stop blurring.
* **Epoch 50-100:** High-frequency textures emerge. You will start seeing distinct vehicle profiles, structural edges of buildings, and varying shades of vegetation.

**Target Metrics:**

* **PSNR (Peak Signal-to-Noise Ratio):** Target > 24.0
* **SSIM (Structural Similarity Index):** Target > 0.75

---

### 🎤 5. DEMO STRATEGY & PITCH DECK ANGLE

For the 3-5 minute demo video, do **not** just show a static gallery of outputs. The judges are from ISRO; they care about mission utility, not just pretty pictures.

**The "So What?" Visualization:**

1. **Split-Screen Slider UI:** Build a Streamlit app where the user uploads an IR image and can smoothly slide a vertical bar left and right to reveal the generated RGB colorization beneath it.
2. **The Killer Feature (Downstream Task):** Include a toggle button labeled `"Run Object Detection"`.
* Show YOLOv10 running on the raw IR image (finding maybe 2-3 bounding boxes with low confidence).
* Show YOLOv10 running on the **Pix2Pix Generated RGB image** (finding 15+ bounding boxes with high confidence).


3. **Talking Point:** *"We didn't just build a colorizer. We built an active enhancement pipeline that bridges the domain gap between night-vision satellite imagery and standard computer vision models, immediately improving ISRO's automated situational awareness."*