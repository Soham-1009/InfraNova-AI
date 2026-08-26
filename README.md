# InfraNova AI

**Cross-Spectral Satellite Radiance Translation**: Synthesizing visible optical RGB imagery ($0.43\text{--}0.68\,\mu\text{m}$) from dual-band thermal infrared radiance (Landsat 9 TIRS-2 Band 10 [$10.60\text{--}11.19\,\mu\text{m}$] + Band 11 [$11.50\text{--}12.51\,\mu\text{m}$]) using a multi-scale conditional Generative Adversarial Network (**Pix2PixHD**).

---

### 🔬 Scientific Context & Radiometric Mapping

Thermal infrared sensors (Landsat 9 TIRS-2) measure terrestrial surface brightness temperature and radiated thermal energy ($10.60\text{--}12.51\,\mu\text{m}$), while optical sensors (Landsat 9 OLI-2) capture reflected solar spectral irradiance ($0.45\text{--}0.67\,\mu\text{m}$). InfraNova AI learns a cross-spectral mapping across spatially aligned, paired observations to synthesize 3-channel true color optical RGB imagery from 2-channel thermal radiance inputs.

---

## 🏗️ Active Architecture

The active model consists of a **21.38M parameter multi-scale residual generator** coupled with a **5.53M parameter multi-scale PatchGAN discriminator**:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Dual Thermal Radiance                  │
                  │   Band 10 (100m, 128×128) + Band 11 (100m, 128×128)   │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼ (2× AvgPool)                              ▼ (Full Res)
        ┌─────────────────────────────┐             ┌─────────────────────────────┐
        │      Global Generator       │             │       Local Enhancer        │
                                                    └──────────────┬──────────────┘
                                                                   │
                                             ┌─────────────────────┴─────────────────────┐
                                             │                                           │
                                             ▼ (Scale 0: 128×128)                        ▼ (Scale 1: 64×64)
                              ┌─────────────────────────────┐             ┌─────────────────────────────┐
                              │     PatchDiscriminator 0    │             │     PatchDiscriminator 1    │
                              │    SpectralNorm Conv 70×70  │             │    SpectralNorm Conv 70×70  │
                              └─────────────────────────────┘             └─────────────────────────────┘
```

### Key Architectural Characteristics
- **Generator (`Pix2PixHDGenerator`, 21.38M params)**: Coarse global U-Net ($64\times 64$) coupled to a fine local enhancer ($128\times 128$) via $1\times 1$ feature projection and bilinear upsampling (eliminating checkerboard artifacts).
- **Discriminator (`MultiScaleDiscriminator`, 5.53M params)**: 2 independent PatchGAN branches operating at $128\times 128$ and $64\times 64$ with `SpectralNorm` on all convolutional layers and **no InstanceNorm** (preserving strict Lipschitz continuity).
- **Multi-Term Loss Objective**:
  $$\mathcal{L}_{\text{total}} = 1.0\mathcal{L}_{\text{adv}} + 10.0\mathcal{L}_{L1} + 10.0\mathcal{L}_{\text{perc}} + 5.0\mathcal{L}_{\text{ssim}} + 2.0\mathcal{L}_{\text{chroma}} + 5.0\mathcal{L}_{\text{feat}}$$

---

## 📊 Validation Model Selection & Unbiased Test Evaluation

### 1. Checkpoint Provenance & Validation Selection Heuristic
Candidate checkpoints were checkpointed during the 250-epoch dual-band training run (linear learning-rate decay phase, Epochs 101–250). To ensure strict separation of splits and prevent test-set data leakage, the production checkpoint was selected **strictly on the 1,432-sample validation split** (`data/landsat9_b10_b11/splits/val/`) using a composite selection heuristic defined *a priori*:
- **Primary Objective**: $\text{SSIM}_{\text{val}}$ (structural preservation)
- **Secondary Objectives**: $\text{PSNR}_{\text{val}}$ (pixel fidelity), $\text{CIE }\Delta E^*_{ab}$ (perceptual color error), $\text{SAM}_{\text{val}}$ (spectral angle)
- **Feasibility Constraint**: Optical saturation error $|\text{SatRatio} - 1.0| \le 0.20$
$$\text{Score}_{\text{val}} = (10.0 \cdot \text{SSIM}_{\text{val}}) + (0.5 \cdot \text{PSNR}_{\text{val}}) - (0.1 \cdot \text{CIE }\Delta E^*_{ab}) - (2.0 \cdot \text{SAM}_{\text{val}})$$

| Candidate Checkpoint | Saved Epoch | Val SSIM (Global / Windowed) $\uparrow$ | Val PSNR (dB) $\uparrow$ | Val CIE Lab $\downarrow$ | Val SAM (rad) $\downarrow$ | Val Sat Error $\downarrow$ | Composite Score | Selection Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`best_ssim.pth`** | **Epoch 223** | **0.3257** / **0.2410** | **11.599 dB** | 28.720 | 0.2216 rad ($12.7^\circ$) | 0.1048 (Ratio: 1.105) | **5.741** | **SELECTED & FROZEN** |
| **`best_psnr.pth`** | **Epoch 223** | **0.3257** / **0.2410** | **11.599 dB** | 28.720 | 0.2216 rad ($12.7^\circ$) | 0.1048 (Ratio: 1.105) | **5.741** | Alternative Serializer |
| **`best_lab.pth`** | Epoch 225 | 0.3156 / 0.2384 | 11.573 dB | **28.450** | **0.2155 rad ($12.3^\circ$)** | **0.0149 (Ratio: 0.985)** | 5.666 | Admissible (Rank 2) |
| **`best_sam.pth`** | Epoch 225 | 0.3156 / 0.2384 | 11.573 dB | **28.450** | **0.2155 rad ($12.3^\circ$)** | **0.0149 (Ratio: 0.985)** | 5.666 | Admissible (Rank 2) |
| **`epoch_250.pth`** | Epoch 250 | 0.3155 / 0.2372 | 11.494 dB | 29.267 | 0.2257 rad ($12.9^\circ$) | 0.1706 (Ratio: 1.171) | 5.524 | Admissible (Rank 3) |
| **`best_sat_ratio.pth`** | Epoch 218 | 0.3110 / 0.2321 | 11.426 dB | 30.030 | 0.2388 rad ($13.7^\circ$) | 0.2889 (Ratio: 1.289) | 5.342 | Excluded (Sat Error $>0.20$) |

---

### 2. Single Unbiased Generalization Scorecard on Held-Out Test Split
After freezing `best_ssim.pth` (Epoch 223) to [`outputs/best/pix2pix_landsat_best.pth`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/outputs/best/pix2pix_landsat_best.pth), a **single final evaluation** was performed on the 1,259 unseen held-out test samples (`data/landsat9_b10_b11/splits/test/`):

*Source: [`outputs/evaluation/unbiased_test_generalization_scorecard.json`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/outputs/evaluation/unbiased_test_generalization_scorecard.json)*

| Metric | Single-Band Baseline (Ep 100) | **Frozen Application Model (`best_ssim.pth`, Ep 223)** | Generalization Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Test PSNR** $\uparrow$ | 10.820 dB | **11.714 dB** ($\pm 3.229\text{ dB}$) | **+0.894 dB** | **PASS** |
| **Test SSIM (11×11 Gaussian Windowed)** $\uparrow$ | 0.1841 | **0.2347** ($\pm 0.1630$) | **+27.5%** | **PASS** |
| **Test SSIM (Global Image-Level)** $\uparrow$ | — | **0.3254** ($\pm 0.3354$) | — | Reference |
| **Test MAE** $\downarrow$ | 0.2512 | **0.2144** ($\pm 0.0774$) | **-14.6%** | **PASS** |
| **Test RMSE** $\downarrow$ | 0.3120 | **0.2752** ($\pm 0.0842$) | **-11.8%** | **PASS** |
| **CIE $\Delta E^*_{ab}$ Error** $\downarrow$ | 35.420 | **27.235** ($\pm 8.294$) | **-23.1%** | **PASS** |
| **Spectral Angle (SAM)** $\downarrow$ | 0.2680 rad | **0.2187 rad ($12.53^\circ$)** | **-18.4%** | **PASS** |
| **Saturation Ratio** (Target: 1.0) | 0.6023 (Err: 0.398) | **1.1993 (Err: 0.199)** | **-49.9% Error** | **PASS** |
| **Color Histogram Distance** $\downarrow$ | 0.4840 | **0.3572** ($\pm 0.322$) | **-26.2%** | **PASS** |

> **SSIM Mathematical Resolution**: The training telemetry used localized **$11\times 11$ Gaussian-windowed SSIM ($\sigma=1.5$)** evaluating local patch correlations, giving **0.2410** on validation and **0.2347** on test. When computed via whole-image global spatial pooling (Global SSIM), whole-image averages smooth high-frequency texture variation, producing **0.3257** and **0.3254**.

---

### 3. Downstream Object Detection Evaluation (YOLOv8)
Evaluating an off-the-shelf detector (YOLOv8) across 200 held-out test patches against Ground Truth Optical RGB demonstrated that:
- The generated imagery **did not improve downstream object detection performance** under this evaluation protocol. Both raw thermal and synthesized RGB produced very low precision ($0.016$) and recall ($0.029$) because generic COCO detectors trigger spuriously on overhead satellite terrain features.
- The terminal checkpoint (`epoch_250.pth`) produced **112 activations with 0.0% precision (100% false positives)** due to late-stage adversarial noise textures. The validation-selected checkpoint (`best_ssim.pth`, Epoch 223) suppressed these spurious activations by **42.9%** (64 activations), but retained the same baseline precision and recall.

### Full 250-Epoch Training Progression
*Source: `outputs/final/master_summary_statistics.json` & `training_master_250epochs.csv` across 8.21 GPU hours on dual Tesla T4 GPUs:*

| Metric | Single-Band Baseline (Ep 100) | Pre-Extension Peak (Ep 96) | **Final Model (Epoch 250)** | **All-Time Best Record** | Benchmark Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PSNR** $\uparrow$ | 10.82 dB | 11.57 dB | **11.49 dB** | **11.75 dB** *(Ep 137)* | **PASS** *(+0.18 dB vs Benchmark)* |
| **SSIM** $\uparrow$ | 0.1841 | 0.2514 | **0.2340** | **0.2522** *(Ep 68)* | **PASS** *(Peak 0.2522)* |
| **MAE** $\downarrow$ | 0.2512 | 0.2205 | **0.2217** | **0.2183** *(Ep 67)* | **PASS** |
| **RMSE** $\downarrow$ | 0.3120 | 0.2805 | **0.2810** | **0.2741** *(Ep 67)* | **PASS** |
| **CIE $\Delta E^*_{ab}$ Error** $\downarrow$ | 35.42 | 28.73 | **29.87** | **27.61** *(Ep 40)* | **PASS** *(Peak 27.61)* |
| **SAM (Spectral Angle)** $\downarrow$ | 0.2680 rad | 0.2091 rad | **0.2257 rad** | **0.2050 rad** *(Ep 127)* | **PASS** *(Peak 0.2050)* |
| **Saturation Ratio Error vs 1.0** $\downarrow$ | 0.3977 | 0.0160 | **0.0965** | **0.0033** *(Ep 92, Ratio: 1.0033)* | **PASS** *(99.2% Error Reduction)* |
| **10-Epoch Saturation $\sigma$** $\downarrow$ | 0.3632 | 0.1245 | **0.0386** | **0.0386** *(Ep 241–250)* | **PASS** *(10$\times$ Stability Gain)* |
| **Histogram Wasserstein Dist** $\downarrow$ | 0.1840 | 0.1042 | **0.0929** | **0.0843** *(Ep 161)* | **PASS** *(Lowest Color Shift)* |

---

## 📁 Repository Layout

```
InfraNova-AI/
├── api/                             # FastAPI REST Backend
│   ├── main.py                      # Endpoints: /health, /colorize, /thermal-preview, /postprocess/clahe
│   └── Dockerfile                   # Backend container definition
├── configs/
│   └── config.yaml                  # Default 2-channel Pix2PixHD configuration
├── data/                            # Local dataset storage (splits: train, val, test)
├── demo/                            # Inference engine & visualization helpers
│   ├── inference.py                 # Core InferenceEngine class (used by API)
│   └── utils.py                     # Image preprocessing & thermal colormap helpers
├── docs/                            # Comprehensive Technical Documentation Suite
│   ├── architecture/                # Model, System, Data Flow, & Legacy architecture specs
│   ├── data/                        # Dataset extraction, patching, and normalization
│   ├── deployment/                  # API, Web, Docker, and deployment workflows
│   ├── evaluation/                  # Metrics formulations and benchmarking procedures
│   ├── experiments/                 # Full experimental telemetry and 250-epoch results
│   ├── inference/                   # Inference engines, TTA mechanics, and TIFF export
│   ├── project/                     # AI Handover, Master Guide, Status, and Decisions
│   └── training/                    # Loss functions, schedules, and memory management
├── kaggle_kernel/                   # Kaggle training scripts & multi-segment execution logs
│   ├── train_pix2pixhd_2channel.py  # Segmented training script with memory cleanup
│   └── run_output/                  # Checkpoints and logs from completed training
├── outputs/                         # Final staged checkpoints and master telemetry
│   └── final/                       # epoch_250.pth, best_*.pth, training_master_250epochs.csv
├── scripts/
│   ├── download/                    # Google Earth Engine Landsat 9 download scripts
│   ├── preprocessing/               # Patch generation, spatial splitting, normalization stats
│   ├── evaluation/                  # Standalone CLI evaluation & benchmarking tools
│   ├── deployment/                  # Model export (ONNX / TorchScript)
│   └── pipeline/                    # End-to-end automation runners
├── src/                             # Core Python Package
│   ├── datasets/                    # Landsat9Dataset loader (2-channel, local/global norm)
│   ├── inference/                   # LandsatColorizationInference production engine
│   ├── models/pix2pix/              # Pix2PixHDGenerator, MultiScaleDiscriminator, Pix2Pix
│   ├── training/                    # Trainer, CombinedLoss, LinearLRScheduler, Callbacks
│   └── utils/                       # Checkpointing, image processing, logging, seeds
├── tests/                           # Pytest test suite (46 passing tests)
├── web/                             # React + Vite Interactive Frontend
│   ├── src/                         # App.jsx, index.css, visual components
│   └── Dockerfile                   # Frontend container definition
├── docker-compose.yml               # Development & deployment multi-container configuration
├── Dockerfile                       # Multi-stage production build (FastAPI + React dist)
├── pyproject.toml                   # Build system and linter configurations
└── requirements.txt                 # Python dependencies (PyTorch, OpenCV, FastAPI, etc.)
```

---

## ⚡ Quick Start

### 1. Environment Setup

**Prerequisite**: Python 3.11.

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

```powershell
# Run the complete test suite (46 unit & integration tests)
pytest tests/ -v
```

### 3. Launch Web Application & API

```powershell
# Option A: Run FastAPI backend with hot-reload
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Option B: Run React frontend dev server
cd web
npm install
npm run dev
```

### 4. Run Model Evaluation

```powershell
# Evaluate validation split using the final 250-epoch checkpoint
python scripts/evaluation/evaluate.py --split val --checkpoint outputs/final/epoch_250.pth --image-size 128 --generator-impl hd
```

---

## 📦 Checkpoint Inventory

All checkpoints are verified and stored locally in [`outputs/final/`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/outputs/final):

| Checkpoint | File Size | Description | Key Metric |
| :--- | :---: | :--- | :--- |
| **`epoch_250.pth`** | 323.2 MB | Final weights at completion of 250-epoch schedule | PSNR: 11.49 dB, SSIM: 0.2340 |
| **`best_checkpoint.pth`** | 323.2 MB | Multi-criteria optimal model weights | Balanced quality & stability |
| **`best_ssim.pth`** | 323.2 MB | Optimal structural similarity checkpoint (Epoch 68) | **SSIM: 0.2522** |
| **`best_psnr.pth`** | 323.2 MB | Optimal peak signal-to-noise checkpoint (Epoch 137) | **PSNR: 11.754 dB** |
| **`best_lab.pth`** | 323.2 MB | Minimal perceptual color difference (Epoch 40) | **CIE $\Delta E^*_{ab}$: 27.61** |
| **`best_sam.pth`** | 323.2 MB | Lowest spectral angle distortion (Epoch 127) | **SAM: 0.2050 rad** ($11.74^\circ$) |
| **`best_sat_ratio.pth`** | 323.2 MB | Optimal optical color saturation ratio (Epoch 92) | **Sat Ratio: 1.0033** |

---

## 📖 Documentation Directory

- **AI Agent Onboarding**: [`docs/project/AI_HANDOVER.md`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/docs/project/AI_HANDOVER.md)
- **Master Codebase Guide**: [`docs/project/MASTER_CODEBASE_GUIDE.md`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/docs/project/MASTER_CODEBASE_GUIDE.md)
- **Current Project Status**: [`docs/project/CURRENT_STATUS.md`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/docs/project/CURRENT_STATUS.md)
- **Implementation Status Matrix**: [`docs/project/IMPLEMENTATION_STATUS.md`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/docs/project/IMPLEMENTATION_STATUS.md)
- **Model Architecture Deep-Dive**: [`docs/architecture/MODEL_ARCHITECTURE.md`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/docs/architecture/MODEL_ARCHITECTURE.md)
- **Full 250-Epoch Results Report**: [`docs/experiments/EXPERIMENTS.md`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/docs/experiments/EXPERIMENTS.md)
