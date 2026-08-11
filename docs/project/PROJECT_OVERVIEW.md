# InfraNova AI — Project Overview

---

## InfraNova AI in 60 Seconds

InfraNova AI takes a **thermal infrared satellite image** (a single grayscale band captured by Landsat 9's TIRS sensor at 100m resolution) and generates a **photorealistic RGB satellite image** that approximates what visible-light cameras would capture. It uses a **Pix2Pix conditional GAN** — a neural network that learns to "colorize" heat maps into natural-looking aerial photos.

**Input**: Grayscale thermal image (128×128 pixels, single channel)  
**Output**: Color RGB image (128×128 pixels, 3 channels)

---

## InfraNova AI in 10 Minutes

### What Problem Does It Solve?

Satellites like Landsat 9 carry thermal sensors that can image Earth's surface day and night, through clouds, and in conditions where visible-light cameras fail. However, thermal images look like blurry grayscale heat maps — they're hard for humans to interpret and lack the color detail needed for applications like land use analysis, urban planning, and disaster monitoring.

InfraNova AI bridges this gap. It takes a thermal image and produces a corresponding visible-light image, enabling users to:
- **Visualize** thermal data in natural color
- **Analyze** land features that are hidden in monochrome thermal imagery
- **Monitor** regions during nighttime or cloud cover using thermal + AI-generated color

### How Does It Work?

1. **Data Pipeline**: Landsat 9 scenes are downloaded from Google Earth Engine. Each scene provides a thermal band (ST_B10) and visible-light bands (SR_B4, SR_B3, SR_B2 for Red, Green, Blue). These are cropped into 128×128 pixel patches and saved as NumPy arrays.

2. **Training**: The Pix2Pix GAN sees thousands of thermal→RGB patch pairs and learns the mapping between them. The generator tries to produce realistic RGB, while the discriminator tries to distinguish generated images from real ones. Five loss functions (adversarial, L1, perceptual, SSIM, chroma) balance realism, accuracy, and color fidelity.

3. **Inference**: Given a new thermal image, the trained generator produces an RGB output in ~50ms on GPU (or ~200ms on CPU). An optional test-time augmentation (TTA) mode averages 4 geometric variants for slightly better quality.

4. **Web Application**: A React frontend lets users drag-and-drop thermal images; a FastAPI backend runs the model and returns the colorized result alongside a thermal preview.

### Major Components

| Component | Technology | Location |
|-----------|-----------|----------|
| Model | PyTorch (Pix2Pix GAN) | `src/models/pix2pix/` |
| Training | Custom trainer with AMP, gradient clipping | `src/training/` |
| Dataset | Landsat 9 via Google Earth Engine | `src/datasets/`, `scripts/` |
| Inference | PyTorch inference engine | `demo/inference.py`, `src/inference/` |
| API | FastAPI | `api/main.py` |
| Frontend | React + Vite | `web/` |
| Deployment | Docker (currently stale) | `Dockerfile`, `docker-compose.yml` |
| Training infra | Kaggle Notebooks (2× Tesla T4) | `notebooks/MAIN.ipynb` |

### Limitations

- SSIM is ~0.43 — outputs are plausible but not photorealistic
- Only uses a single thermal band (TIR) as input
- Fixed 128×128 resolution
- Model may produce brownish/desaturated outputs (val_sat_ratio ~0.44)
- Not suitable for pixel-perfect remote sensing; this is a visualization tool

---

## InfraNova AI Technically

### Architecture

- **Generator**: `GeneratorUNetDynamic` — a U-Net encoder-decoder. For `image_size=128`, it has `depth=7` (since `log2(128) = 7`). Feature channels: `[64, 128, 256, 512, 512, 512, 512]`. Uses `InstanceNorm2d`, `LeakyReLU(0.2)` in encoder, `ReLU` and `Dropout(0.5)` on first 3 decoder blocks. Bilinear upsample + Conv2d instead of transposed convolution (prevents checkerboard artifacts).

- **Discriminator**: `PatchDiscriminator` — 70×70 receptive field PatchGAN. All convolutional layers use `SpectralNorm` (no InstanceNorm). Input: concatenated IR + RGB (4 channels). Output: patch scores.

- **Loss**: `CombinedLoss` = `λ_adv × GAN(BCE) + λ_l1 × L1 + λ_perc × VGGPerceptual + λ_ssim × (1-SSIM) + λ_chroma × ChromaL1`

### Training Environment

- Hardware: 2× Tesla T4 (Kaggle)
- Batch size: 128
- AMP: enabled
- DataParallel: multi-GPU
- Total epochs trained: 226 (out of 250)
- Training time: ~6h 12m (across resumable sessions)

### Dataset

- Source: Landsat 9 Level-2 Surface Reflectance + Surface Temperature
- Bands: TIR = `ST_B10` (100m), RGB = `SR_B4/B3/B2` (30m, resampled to 100m)
- Patches: 128×128 pixels, stored as `.npy` files
- Splits: Region-based (preventing spatial data leakage)
- Normalization: Per-sample percentile stretching (p2, p98) → [-1, 1]

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| ML Framework | PyTorch ≥ 2.0 |
| Image Processing | OpenCV, Pillow, rasterio, tifffile |
| Satellite Data | Google Earth Engine (earthengine-api, geemap) |
| Web Backend | FastAPI + Uvicorn |
| Web Frontend | React 19 + Vite 6 |
| Code Quality | Ruff, Black, isort, mypy, pre-commit |
| Testing | pytest |
| Deployment | Docker |
| Training Infra | Kaggle (GPU notebooks) |
