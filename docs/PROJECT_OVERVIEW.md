# InfraNova AI — Project Overview

> **Transforming Invisible Thermal Heat Maps into Visible Satellite Imagery Using AI**

---

## 🎯 The Problem

Satellites like **Landsat 9** capture images of Earth using thermal infrared (TIR) sensors. These sensors detect **heat radiation** instead of visible light, producing grayscale images that show temperature patterns — hot rooftops appear bright, cool water appears dark.

The problem? **Humans and computer vision tools struggle to interpret these thermal images.** A military analyst, urban planner, or disaster responder looking at a thermal image cannot easily distinguish a road from a building, a forest from farmland, or a river from a shadow. Everything looks like a blurry gray heatmap.

**What if we could teach an AI to "colorize" these thermal images — converting them into realistic RGB satellite photos?**

That's exactly what InfraNova AI does.

---

## 💡 The Solution

InfraNova AI is a **deep learning pipeline** that takes a single-channel thermal infrared image (grayscale) and generates a **3-channel RGB satellite image** that looks like it was captured by a normal visible-light camera.

```
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│                  │         │                  │         │                  │
│   Thermal Input  │  ────►  │   AI Model       │  ────►  │   RGB Output     │
│   (Grayscale)    │         │   (Pix2Pix GAN)  │         │   (Full Color)   │
│                  │         │                  │         │                  │
└──────────────────┘         └──────────────────┘         └──────────────────┘
     1 channel                  Generator +                  3 channels
     (heat map)               Discriminator               (realistic photo)
```

### Why This Matters
- **Night Vision:** Thermal sensors work in complete darkness. Our AI can generate daytime-quality RGB images from nighttime thermal data.
- **Cloud Penetration:** Thermal radiation passes through thin clouds. This means usable imagery even during overcast conditions.
- **Disaster Response:** Emergency teams can get interpretable satellite views of flood zones, wildfires, or earthquake damage — even at night.
- **Urban Planning:** City planners can visualize heat islands and infrastructure from thermal data with much greater clarity.

---

## 🏗️ How It Works (Technical Overview)

### The AI Architecture: Pix2Pix Conditional GAN

We use a **Pix2Pix** model, which is a type of Generative Adversarial Network (GAN) specifically designed for image-to-image translation tasks.

```
                    ┌─────────────────────────────────┐
                    │         GENERATOR                │
                    │     (Dynamic U-Net)              │
                    │                                  │
  Thermal Input ──► │  Encoder ──► Bottleneck ──► Decoder ──► RGB Output
  (128×128, 1ch)    │     │                        ▲       │  (128×128, 3ch)
                    │     └──── Skip Connections ──┘       │
                    └─────────────────────────────────┘
                                    │
                              Generated RGB
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │       DISCRIMINATOR              │
                    │     (PatchGAN 70×70)             │
                    │                                  │
                    │  "Is this pair (thermal, RGB)    │
                    │   real or generated?"            │
                    └─────────────────────────────────┘
```

**Two neural networks compete against each other:**

| Component | Role | Architecture |
|-----------|------|-------------|
| **Generator** | Creates fake RGB images from thermal input | Dynamic U-Net (depth auto-adapts to input size) |
| **Discriminator** | Tries to distinguish real RGB from generated RGB | PatchGAN (classifies 70×70 patches as real/fake) |

The Generator gets better at fooling the Discriminator, and the Discriminator gets better at catching fakes. This adversarial competition drives the Generator to produce increasingly realistic images.

### The Loss Function (5 Components)

The Generator is trained with a composite loss that balances multiple objectives:

| Loss Component | Weight | Purpose |
|----------------|--------|---------|
| **Adversarial (BCE)** | 1.0 | Forces outputs to look realistic |
| **L1 (Pixel)** | 10.0 | Ensures pixel-level accuracy |
| **Perceptual (VGG19)** | 10.0 | Preserves high-level features (roads, buildings, vegetation) |
| **SSIM** | 5.0 | Maintains structural similarity |
| **Chroma** | 2.0 | Ensures color consistency |

---

## 📊 The Dataset

### Source: Landsat 9 Satellite (via Google Earth Engine)

| Property | Value |
|----------|-------|
| **Satellite** | Landsat 9 (NASA/USGS) |
| **Input Band** | ST_B10 (Thermal Infrared, 10.9 µm) |
| **Target Bands** | SR_B2, SR_B3, SR_B4 (Blue, Green, Red) |
| **Patch Size** | 128 × 128 pixels |
| **Total Training Samples** | ~13,384 paired patches |
| **Validation Samples** | ~1,630 paired patches |
| **Geographic Diversity** | 50 regions across multiple biomes |

### Biome Coverage

| Biome Type | Thermal Characteristics |
|------------|------------------------|
| **Urban** | Heat islands → warm signatures even at night |
| **Desert** | Extreme heating by day, rapid cooling at night |
| **Tropical Forest** | Transpiration cooling → low thermal signal |
| **Cold/Snow** | Low temperatures, high emissivity |
| **Coastal** | Sharp land-water temperature boundaries |

### Data Pipeline

```
Google Earth Engine API
        │
        ▼
  Download raw Landsat 9 bands (B2, B3, B4, B10)
        │
        ▼
  Organize into per-region folders
        │
        ▼
  Build 128×128 paired patches (thermal ↔ RGB)
        │
        ▼
  Filter anomalous data (NoData, blank tiles)
        │
        ▼
  Region-level train/val/test split
  (prevents geographic leakage)
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| **Framework** | PyTorch |
| **Image Size** | 128 × 128 |
| **Batch Size** | 128 |
| **Epochs** | 250 (with linear LR decay from epoch 230) |
| **Learning Rate** | 0.0002 (Adam optimizer) |
| **Mixed Precision** | Enabled (AMP) — 2× speedup |
| **Hardware** | 2× NVIDIA Tesla T4 GPUs (Kaggle) |
| **Multi-GPU** | DataParallel (generator + discriminator parallelized individually) |
| **Early Stopping** | Patience of 40 epochs on validation SSIM |

---

## 📈 Early Training Results (First 3 Epochs)

| Epoch | Generator Loss | Discriminator Loss | Train SSIM | Val PSNR | Val SSIM |
|-------|---------------|-------------------|------------|----------|----------|
| 1/10 | 54.12 | 0.52 | 0.7727 | 11.36 | 0.2330 |
| 2/10 | 48.78 | 0.42 | 0.7231 | 11.84 | 0.3108 |
| 3/10 | 48.34 | 0.40 | 0.6997 | 12.03 | 0.3382 |

**Key Insight:** Validation SSIM improved from 0.23 → 0.34 in just 3 epochs, proving the model is successfully learning the thermal-to-RGB mapping.

### Target Benchmarks

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| **SSIM** | ≥ 0.20 | 0.25 | 0.30+ |
| **PSNR** | ≥ 18 dB | 22 dB | 25 dB |
| **LPIPS** | ≤ 0.35 | ≤ 0.25 | ≤ 0.20 |
| **FID** | ≤ 80 | ≤ 60 | ≤ 50 |

---

## 🧠 Why This Is Hard

Unlike converting a black-and-white photo to color (where brightness correlates with color), **thermal images have almost zero correlation with visible color**.

A temperature reading of 305K (32°C) could be:
- Dark **asphalt** road (gray)
- Light **concrete** building (white)
- **Rocky desert** (brown)
- **Dry soil** (beige)

The AI must learn to use **spatial context** (surrounding patterns, textures, edges) to infer what the most plausible color should be. This makes TIR-to-RGB one of the hardest image translation problems in remote sensing.

> **Literature benchmark:** State-of-the-art models achieve SSIM of only 0.22–0.28 for TIR-to-RGB, compared to 0.30–0.35 for the much easier NIR-to-RGB task.

---

## 🗂️ Project Structure

```
InfraNova-AI/
├── configs/                  # YAML training configurations
│   ├── config.yaml           #   Production (250 epochs)
│   └── config_smoke.yaml     #   Smoke test (1 epoch)
├── src/                      # Core source code
│   ├── datasets/             #   Landsat 9 data loader
│   ├── models/pix2pix/       #   Generator, Discriminator, Pix2Pix wrapper
│   ├── training/             #   Training loop, losses, callbacks
│   ├── inference/            #   Production inference engine
│   ├── evaluation/           #   Metric computation (PSNR, SSIM, LPIPS)
│   └── utils/                #   Checkpointing, logging, helpers
├── scripts/                  # Pipeline automation scripts
│   ├── download/             #   Google Earth Engine export
│   ├── preprocessing/        #   Patch building, splitting, filtering
│   ├── evaluation/           #   Benchmarking & validation
│   └── deployment/           #   ONNX export, model cards, batch inference
├── demo/                     # Streamlit web application
│   └── streamlit_app.py      #   Upload thermal → view generated RGB
├── docs/                     # Technical documentation
├── tests/                    # Unit and integration tests
└── data/                     # Dataset (not committed to Git)
```

---

## 🖥️ Streamlit Demo Application

The project includes a **web-based demo** built with Streamlit that allows users to:

1. **Upload** a thermal infrared image
2. **Generate** an RGB colorized output using the trained model
3. **Compare** input vs. output side-by-side
4. **Download** the generated RGB image
5. Toggle **test-time augmentation** for improved quality
6. Switch between **light and dark mode**

---

## 🚀 Deployment Options

| Method | Use Case |
|--------|----------|
| **Streamlit App** | Interactive web demo for stakeholders |
| **Docker Container** | Portable deployment on any server |
| **ONNX Export** | Cross-platform inference (C++, mobile, edge devices) |
| **TorchScript** | Optimized PyTorch production serving |
| **Batch Inference** | Process thousands of thermal tiles automatically |

---

## 🔮 Future Enhancements

1. **Super-Resolution:** Joint 2× upscaling + colorization (200m → 100m resolution enhancement)
2. **Object Detection Integration:** Run YOLOv8 on colorized output to detect buildings, roads, vehicles
3. **Multi-Temporal Analysis:** Process thermal time-series to track urban heat islands over time
4. **Real-Time Inference:** Optimize for edge deployment on satellite ground stations

---

## 📚 Tech Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning** | PyTorch, torchvision |
| **Model** | Pix2Pix (Dynamic U-Net + PatchGAN) |
| **Perceptual Loss** | VGG19 (pretrained on ImageNet) |
| **Data Source** | Google Earth Engine API |
| **Training** | Kaggle (2× T4 GPUs), AMP, DataParallel |
| **Demo** | Streamlit |
| **Containerization** | Docker + Docker Compose |
| **Metrics** | PSNR, SSIM, LPIPS, FID |
| **Version Control** | Git |

---

*Built by Soham Deshpande — InfraNova AI*
