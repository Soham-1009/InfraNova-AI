# InfraNova AI

**Cross-Spectral Satellite Radiance Translation**: Synthesizing visible optical RGB imagery ($0.43$--$0.68\,\mu\text{m}$) from dual-band thermal infrared radiance (Landsat 9 TIRS-2 Band 10 [$10.60$--$11.19\,\mu\text{m}$] + Band 11 [$11.50$--$12.51\,\mu\text{m}$]) using a compact Global ResNet conditional Generative Adversarial Network (**Pix2PixHD-style LSGAN**).

---

### 🔬 Scientific Context & Radiometric Mapping

Thermal infrared sensors (Landsat 9 TIRS-2) measure terrestrial surface brightness temperature and radiated thermal energy ($10.60$--$12.51\,\mu\text{m}$), while optical sensors (Landsat 9 OLI-2) capture reflected solar spectral irradiance ($0.45$--$0.67\,\mu\text{m}$). InfraNova AI learns a cross-spectral mapping across spatially aligned, paired observations to synthesize 3-channel true color optical RGB imagery from 2-channel thermal radiance inputs.

---

## 🚦 Production Checkpoint

The active production serving model is **Exp9**:

- **Active Production Checkpoint**: [`outputs/best/pix2pix_landsat_best.pth`](outputs/best/pix2pix_landsat_best.pth)
- **Checkpoint SHA-256**: `71bbda3f31b85e7e741b26d5ce0ff398a0394c7dd4ef6ff5452f31f7e0400382`
- **Generator Architecture**: `Pix2PixHDGlobalResNetGenerator`
- **Active Generator Parameters**: **11,369,795**
- **Parameter Reduction vs. Legacy**: Reduced by **10,013,443 parameters (46.83%)** compared to the previous 21,383,238-parameter generator.
- **Rollback Safety**: The previous production checkpoint is preserved at `outputs/best/pix2pix_landsat_backup_20260912_231938.pth` (SHA-256: `4604d36d07a0fb4c0696a53040c17004084068c51cc745a23f69b76c8baf6aa8`), with full provenance recorded in [`outputs/best/rollback_metadata.json`](outputs/best/rollback_metadata.json).

---

## 🏗️ Active Architecture

The production architecture employs a **Pix2PixHD-style Global ResNet Generator** (11.37M parameters) paired with a **MultiScaleDiscriminator** (5.53M parameters):

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Dual Thermal Radiance                  │
                  │   Band 10 (100m, 128×128) + Band 11 (100m, 128×128)   │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼ (7×7 Conv, ReflectionPad, 64ch)
                                  [Initial Convolution]
                                             │
                                             ▼ (Stride 2 Conv, 128ch, 64×64)
                                    [Downsampling Stage 1]
                                             │
                                             ▼ (Stride 2 Conv, 256ch, 32×32)
                                    [Downsampling Stage 2]
                                             │
                                             ▼ (9 Residual Blocks, 256ch, 32×32)
                                   [ResNet Bottleneck 9×]
                                             │
                                             ▼ (Stride 2 TransposedConv, 128ch, 64×64)
                                     [Upsampling Stage 1]
                                             │
                                             ▼ (Stride 2 TransposedConv, 64ch, 128×128)
                                     [Upsampling Stage 2]
                                             │
                                             ▼ (7×7 Conv, ReflectionPad, 3ch)
                                    [Output Convolution]
                                             │
                                             ▼ (Tanh Activation)
                                  [Synthesized Optical RGB]
                                             │
                        ┌────────────────────┴────────────────────┐
                        │                                         │
                        ▼ (Scale 0: 128×128)                      ▼ (Scale 1: 64×64)
         ┌─────────────────────────────┐           ┌─────────────────────────────┐
         │    PatchDiscriminator 0     │           │    PatchDiscriminator 1     │
         │   SpectralNorm Conv 70×70   │           │   SpectralNorm Conv 70×70   │
         └─────────────────────────────┘           └─────────────────────────────┘
```

### Key Architectural Characteristics

- **Generator (`Pix2PixHDGlobalResNetGenerator`, 11,369,795 params)**:
  - Input: 2 channels (Landsat 9 B10 + B11 thermal radiance)
  - Output: 3 channels (visible optical RGB)
  - $7\times 7$ reflection-padded input convolution
  - 2 strided downsampling stages reaching $256$ feature channels at $32\times 32$
  - 9 residual bottleneck blocks with reflection padding, instance normalization, and ReLU
  - 2 strided upsampling stages with transposed convolutions
  - $7\times 7$ reflection-padded output convolution with $\text{Tanh}$ activation mapping to $[-1.0, 1.0]$
- **Discriminator (`MultiScaleDiscriminator`, 5.53M params)**: 2 independent PatchGAN branches operating at $128\times 128$ and $64\times 64$ with `SpectralNorm` on all convolutional layers and no `InstanceNorm` (preserving strict Lipschitz continuity).
- **Adversarial Formulation**: Least Squares GAN (**LSGAN** MSE loss).
- **Multi-Term Production Loss Objective**:
  $$\mathcal{L}_{\text{total}} = 1.0\,\mathcal{L}_{\text{adv}} + 10.0\,\mathcal{L}_{L1} + 5.0\,\mathcal{L}_{\text{perc}} + 5.0\,\mathcal{L}_{\text{ssim}} + 2.0\,\mathcal{L}_{\text{chroma}} + 0.05\,\mathcal{L}_{\text{sat}}$$
  *(Feature-matching loss is strictly disabled with $\lambda_{\text{feat}} = 0.0$.)*

---

## 🔬 Scientific Research History & Ablation Suite

Across the research cycle, nine hypothesis-driven experiments were executed:

1. **Exp1 — Perceptual Loss Recovery**: Re-enabled VGG-19 perceptual loss ($\lambda_{\text{perc}} = 5.0$) after catastrophic blur with L1-only loss.
2. **Exp2 — Feature-Matching Ablation**: Set $\lambda_{\text{feat}} = 0.0$, eliminating generator multi-scale intermediate layer matching with no regression.
3. **Exp3 — Chroma Penalty Increase**: Evaluated $\lambda_{\text{chroma}} = 4.0$ to mitigate false spectral casting.
4. **Exp4 — Reduced Perceptual Weight**: Evaluated $\lambda_{\text{perc}} = 3.0$ for high-frequency detail balance.
5. **Exp5 — Saturation Loss**: Introduced differentiable optical saturation regularization ($\lambda_{\text{sat}} = 0.05$) to eliminate low-saturation wash.
6. **Exp6 — BCE $\to$ LSGAN Transition**: Replaced binary cross-entropy with Least Squares GAN loss to stabilize adversarial training dynamics.
7. **Exp7 — LSGAN + Saturation Loss**: Unified LSGAN objective with saturation regularizer ($\lambda_{\text{sat}} = 0.05$).
8. **Exp8 — Increased Saturation Penalty**: Evaluated $\lambda_{\text{sat}} = 0.10$ (over-saturated vegetation textures).
9. **Exp9 — Global ResNet Generator**: Replaced multi-scale U-Net with compact Global ResNet generator ($11.37\text{M}$ params, `gan_mode = lsgan`, `lambda_sat = 0.05`), achieving state-of-the-art results across structural, pixel, and spectral metrics.

**Conclusion**: **Exp9** proved the strongest candidate across the research cycle and was promoted to production.

### Training-Budget Study & Conclusion

- **Exp9 Baseline**: Trained with Adam ($	ext{lr}=2\times 10^{-4}, \beta_1=0.5, \beta_2=0.999$, batch size 64, image size 128, local normalization, `lambda_sat=0.05`). Optimal validation performance was reached at **Epoch 46**.
- **Long Exp9 Continuation**: The training budget was expanded to 250 epochs with patience 100 on GPU. The run triggered early stopping at epoch 130 (84 non-improving epochs); evaluations showed degraded held-out test performance (PSNR: 13.584 dB, SSIM: 0.4493, Saturation ratio: 1.1805) due to late-stage spectral overfitting.
- **Conclusion**: Long Exp9 was formally rejected. Further increases in training duration are exhausted and not part of the production configuration.

---

## 📊 Final Held-Out Test Evaluation

Standardized evaluation across the **1,259 held-out test samples** (`data/landsat9_b10_b11/splits/test/`):

| Metric | Previous Production | Exp9 (Active Production) | Status |
| :--- | :---: | :---: | :---: |
| **PSNR (dB)** $\uparrow$ | 11.7138 | **13.7447** | **+2.0309 dB** |
| **SSIM** $\uparrow$ | 0.3254 | **0.4502** | **+38.35%** |
| **MAE** $\downarrow$ | 0.2144 | **0.1723** | **-19.64%** |
| **RMSE** $\downarrow$ | 0.2752 | **0.2217** | **-19.44%** |
| **SAM (rad)** $\downarrow$ | 0.2187 | **0.1968** | **-10.01%** |
| **SAM (degrees)** $\downarrow$ | 12.53° | **11.28°** | **-1.25°** |
| **CIE Lab $\Delta E^*_{ab}$** $\downarrow$ | 27.2355 | **22.5706** | **-17.13%** |
| **Saturation Ratio** (target 1.0) | 1.1993 | **0.9340** | **+66.88% closer** |
| **Saturation Deviation** $|R - 1.0|$ $\downarrow$ | 0.1993 | **0.0660** | **-66.88%** |
| **Histogram Distance** $\downarrow$ | **0.3572** | 0.4607 | Known trade-off (+0.1035) |

> **Analysis**: Exp9 improves the principal structural, pixel, spectral, perceptual, and saturation metrics. Histogram distance remains worse than the previous production model.

---

## 🎯 Downstream Object Detection (YOLOv8) Evaluation

Downstream object detection performance evaluated with YOLOv8 on the 1,259 held-out test set benchmark:

| Metric | Previous Production | Exp9 (Active Production) | Delta |
| :--- | :---: | :---: | :---: |
| **F1 @ IoU 0.25** $\uparrow$ | 0.2992 | **0.3711** | **+0.0719** |
| **F1 @ IoU 0.50** $\uparrow$ | 0.2835 | **0.3643** | **+0.0808** |
| **Precision @ 0.25** $\uparrow$ | 0.4176 | **0.4219** | **+0.0043** |
| **Recall @ 0.25** $\uparrow$ | 0.2331 | **0.3313** | **+0.0982** |
| **Mean Matched IoU** $\uparrow$ | 0.9977 | **0.9980** | **+0.0003** |

Exp9 improved downstream YOLO detection F1 scores across both IoU thresholds.

---

## 🚀 Production Inference Pipeline

The deployment pipeline performs sliding-window inference across arbitrary-dimension satellite rasters:

```
Band 10 + Band 11 Dual Thermal Input
                ↓
    128×128 Overlapping Tiles
                ↓
   Exp9 ResNet Generator (11.37M)
                ↓
  2D Cosine / Hann Window Blending
                ↓
    Seamless Full Optical RGB
```

### Validated Whole-Raster Benchmark
- **Input Scene**: $1002 \times 1001$ pixels dual-band GeoTIFF
- **Tile Configuration**: $128 \times 128$ tile size, $32\text{ px}$ overlap, $96\text{ px}$ stride
- **Total Tiles Processed**: 121 tiles
- **End-to-End Latency**: **1.740 s**
- **Throughput**: **69.5 tiles/sec**
- **Peak GPU VRAM**: **263.39 MB**
- **Boundary Quality**: Continuous, seam-free blending across all tile boundaries

---

## 🔌 API & Application Services

### FastAPI Backend (`api/`)
- **Lifecycle**: The production model is loaded once into memory on application startup.
- **Header**: All `/colorize` responses include the telemetry header `X-Model: pix2pix-landsat-exp9-resnet`.
- **Endpoints**:
  - `GET  /health` — Service readiness, model status, and compute device
  - `POST /colorize` — Full cross-spectral translation on uploaded image or raster
  - `POST /thermal-preview` — False-color thermal visualization
  - `POST /postprocess/clahe` — Adaptive contrast enhancement postprocessing

### Interactive Web Dashboard (`web/`)
Built with React + Vite:
- Multi-format file ingestion: PNG/JPEG satellite images, dual-band `.npy` arrays, and GeoTIFF files
- Preset validation scenes (Accra, Abu Dhabi, etc.)
- Before/after comparison visual slider with synchronized zoom
- Test-Time Augmentation (TTA) toggles (horizontal, vertical, rotation transforms)
- Real-time inference latency and memory telemetry
- CLAHE postprocessing controls and single-click high-resolution PNG export

---

## 📁 Repository Structure

```
InfraNova-AI/
├── api/             # FastAPI REST backend and container definitions
├── configs/         # Experiment configurations (config.yaml, exp1–exp9)
├── data/            # Landsat-9 dual-band dataset storage and spatial splits
├── demo/            # InferenceEngine, CLI runners, and colormap helpers
├── docs/            # Architecture, deployment, training, and evaluation documentation
├── outputs/         # Staged production model, historical records, and reports
│   ├── best/        # Active production checkpoint (pix2pix_landsat_best.pth) and rollback backup
│   ├── exp1/ ... exp9/ # Historical scientific experiment outputs and checkpoints
│   └── final/       # Historical reference checkpoints and master progression logs
├── reports/         # Visual inspection panels and final adjudication reports
├── scripts/         # Reusable evaluation, deployment, preprocessing, and download tools
├── tests/           # Automated pytest test suite
└── web/             # React + Vite interactive satellite colorization frontend
```

*Note: Large model weights (`.pth`), checkpoints, and raw satellite datasets are excluded from Git tracking.*

---

## ⚡ Quick Start

### Option A: macOS & Linux (Bash / Zsh)

#### 1. Prerequisites & System Packages

- **Python**: 3.11 (`python3 --version`)
- **Node.js**: 18+ and npm (`node -v`, `npm -v`)
- **Linux (Ubuntu/Debian)** system libraries:
  ```bash
  sudo apt update && sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
  ```
- **macOS (Homebrew)**:
  ```bash
  brew install python@3.11 node
  ```

#### 2. Environment Setup

```bash
git clone https://github.com/Soham-1009/InfraNova-AI.git
cd InfraNova-AI

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Run Test Suite (85 Tests)

```bash
pytest tests/ -v
```

#### 4. Launch Application

```bash
# Terminal 1: Run FastAPI backend with hot-reload (port 8000)
source venv/bin/activate
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Run React + Vite frontend development server (port 5173)
cd web
npm install
npm run dev
```

#### 5. Run Model Evaluation

```bash
# Run benchmark evaluation on active production Exp9 ResNet checkpoint
python scripts/evaluation/run_single_candidate.py --checkpoint outputs/best/pix2pix_landsat_best.pth
```

---

### Option B: Windows (Command Prompt — `cmd.exe`)

#### 1. Prerequisites

- **Python**: 3.11 (`python --version`)
- **Node.js**: 18+ and npm (`node -v`, `npm -v`)
- **Git for Windows**

#### 2. Environment Setup

```cmd
git clone https://github.com/Soham-1009/InfraNova-AI.git
cd InfraNova-AI

python -m venv venv
venv\scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Run Test Suite (85 Tests)

```cmd
pytest tests/ -v
```

#### 4. Launch Application

```cmd
:: Terminal 1: Run FastAPI backend with hot-reload (port 8000)
venv\scripts\activate
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

:: Terminal 2: Run React + Vite frontend development server (port 5173)
cd web
npm install
npm run dev
```

#### 5. Run Model Evaluation

```cmd
:: Run benchmark evaluation on active production Exp9 ResNet checkpoint
python scripts/evaluation/run_single_candidate.py --checkpoint outputs/best/pix2pix_landsat_best.pth
```

---

### Option C: Windows (PowerShell)

#### 1. Environment Setup

```powershell
git clone https://github.com/Soham-1009/InfraNova-AI.git
cd InfraNova-AI

python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Run Test Suite (85 Tests)

```powershell
pytest tests/ -v
```

#### 3. Launch Application

```powershell
# Terminal 1: Run FastAPI backend with hot-reload (port 8000)
.\venv\Scripts\Activate.ps1
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Run React + Vite frontend development server (port 5173)
cd web
npm install
npm run dev
```

#### 4. Run Model Evaluation

```powershell
# Run benchmark evaluation on active production Exp9 ResNet checkpoint
python scripts/evaluation/run_single_candidate.py --checkpoint outputs/best/pix2pix_landsat_best.pth
```

---

The web interface is served at `http://localhost:5173`, with interactive OpenAPI documentation at `http://localhost:8000/docs`.

---

## 📖 Documentation Directory

- **Current Project Status**: [`docs/project/CURRENT_STATUS.md`](docs/project/CURRENT_STATUS.md)
- **Master Codebase Guide**: [`docs/project/MASTER_CODEBASE_GUIDE.md`](docs/project/MASTER_CODEBASE_GUIDE.md)
- **AI Agent Onboarding**: [`docs/project/AI_HANDOVER.md`](docs/project/AI_HANDOVER.md)
- **Model Architecture Deep-Dive**: [`docs/architecture/MODEL_ARCHITECTURE.md`](docs/architecture/MODEL_ARCHITECTURE.md)
- **Full Experimental Telemetry**: [`docs/experiments/EXPERIMENTS.md`](docs/experiments/EXPERIMENTS.md)

---

## 🏁 Final Project Status

```
Research cycle: COMPLETE
Model selection: COMPLETE
Production promotion: COMPLETE
Production model: Exp9
Deployment validation: COMPLETE
Long-training study: COMPLETE
```

**Exp9 is the active production model.**
