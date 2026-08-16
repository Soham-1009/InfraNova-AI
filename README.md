# InfraNova AI

**[📖 Read the Master Codebase Guide](docs/project/MASTER_CODEBASE_GUIDE.md)** for complete, truth-based technical documentation, architecture diagrams, and developer onboarding guides.

InfraNova AI converts Landsat 9 thermal infrared imagery into RGB-like satellite imagery using a Pix2Pix conditional GAN.

The project is designed around a simple idea: thermal images show heat patterns, while most humans and computer vision tools are easier to work with when imagery looks like normal RGB satellite data. The model learns from paired Landsat 9 thermal and RGB bands, then generates a plausible RGB view from a single thermal input.

## What This Project Does

- Downloads Landsat 9 thermal and RGB bands from Google Earth Engine.
- Organizes raw `.tif` exports into region folders.
- Builds paired 128×128 training patches, filtering out anomalous data (NoData/Blank).
- Generates reproducible dataset fingerprints and experiment telemetry.
- Trains a Pix2Pix GAN (with Dynamic U-Net generator) for TIR-to-RGB colorization.
- Supports Kaggle GPU training with a dedicated smoke-test configuration.
- Runs inference from a checkpoint with optional test-time augmentation.
- Provides a **React + FastAPI web app** for uploading thermal images and viewing generated RGB output.

## Important Limitation

Thermal infrared data does not contain true color information. The model is not recovering exact original colors from physics. It learns likely visual patterns from training data.

For example, a bright thermal area may be road, rooftop, dry soil, or rock. The generated RGB output should be treated as a plausible reconstruction, not guaranteed ground truth.

## Architecture

```text
Landsat 9 bands
    |
    |-- SR_B2, SR_B3, SR_B4  -> RGB target
    |-- ST_B10              -> thermal input
    |
    v
Patch preparation (128×128)
    |
    v
Pix2Pix model
    |
    |-- Dynamic U-Net generator  -> creates RGB image
    |-- PatchGAN discriminator   -> checks realism
    |
    v
RGB-like satellite output
```

## Repository Layout

```text
configs/
  config.yaml                 Production training configuration (500 epochs)

src/
  datasets/                   Landsat 9 dataset loader
  models/pix2pix/             Dynamic U-Net generator, PatchGAN discriminator, Pix2Pix wrapper
  training/                   Losses, trainer, callbacks, scheduler
  inference/                  Production inference engine
  losses/                     Loss function definitions
  evaluation/                 Metric computation
  detection/                  Object detection integration
  utils/                      Checkpoint, logging, and helper utilities

scripts/
  download/                   Earth Engine export scripts
  preprocessing/              Patch building, splitting, filtering, normalization
  evaluation/                 Evaluation, benchmarking, dataset validation, inference testing
  deployment/                 Model card generation, batch inference, ONNX/TorchScript export
  training/                   Training launch scripts
  pipeline/                   End-to-end pipeline orchestration

api/
  main.py                     FastAPI backend (REST API for inference)

web/                          React frontend (Vite)
  src/
    App.jsx                   Main application component
    index.css                 Design system and styles

demo/
  inference.py                Inference engine wrapper
  utils.py                    Preprocessing and display helpers

docs/
  project/                    Comprehensive truth-based documentation system
    MASTER_CODEBASE_GUIDE.md  Start here! Master index for all documents
    AI_HANDOVER.md            State document for AI agent onboarding
    ARCHITECTURE.md           System architecture and diagrams
    FLOW.md                   Execution flow traces
    ... (and 20+ other technical deep-dives)

notebooks/
  MAIN.ipynb                  Kaggle training notebook

tests/                        Unit and integration tests
requirements.txt              Python dependencies
```

## Setup

**Prerequisite:** Python 3.11 is required.

To create and activate a new virtual environment:

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Or run commands directly through the environment:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### PyTorch Install Options

`requirements.txt` lists the PyTorch packages, but you should install the build that matches your hardware before the rest of the dependencies.

CPU-only:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

CUDA 12.1:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

On Kaggle, PyTorch is pre-installed. Skip the PyTorch install step and install only the remaining dependencies.

### Troubleshooting: C++ Build Tools Error

If you see this error while running `pip install -r requirements.txt`:
`error: Microsoft Visual C++ 14.0 or greater is required. Failed building wheel for stringzilla`

This happens because `pip` cannot find a pre-compiled Windows binary for `stringzilla` and tries to compile it from source, which requires a C++ compiler. 

**Recommended Fix (Permanent):**
1. Download [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Run the installer and check **"Desktop development with C++"**.
3. Install, restart your terminal, activate your `venv`, and run the `pip install` command again.

**Quick Workaround (Bypass Compilation):**
If you want to force pip to only use pre-compiled binaries (which will skip compilation but may fail if no binary exists at all), run this in your `venv` before installing requirements:
```powershell
pip install --only-binary=:all: stringzilla
```

## Data Pipeline

### 1. Export Landsat 9 data

```powershell
venv\Scripts\python.exe scripts\download\download_landsat9.py
```

This starts Earth Engine export tasks for Landsat 9 bands:

- `SR_B2`: blue
- `SR_B3`: green
- `SR_B4`: red
- `ST_B10`: thermal infrared

The files are exported to Google Drive.

### 2. Organize downloaded files

After downloading the Drive exports locally:

```powershell
venv\Scripts\python.exe scripts\preprocessing\organize_files.py --source path\to\downloaded\folder
```

This creates per-region folders under:

```text
data/landsat9/input/
```

### 3. Build patches

```powershell
venv\Scripts\python.exe scripts\preprocessing\process_landsat_patches.py
```

This creates 128×128 paired samples under:

```text
data/landsat9/patches/
```

Each sample contains:

- `tir_200m.npy`
- `tir_100m.npy`
- `rgb_100m.npy`

### 4. Create splits

```powershell
venv\Scripts\python.exe scripts\preprocessing\split_patches.py --overwrite
```

The split is region-level. Patches from the same region are kept in only one of train, validation, or test to avoid evaluation leakage. The splitter also keeps small datasets usable by avoiding empty train/val partitions when there are only a few regions.

## Training

Training is configured in:

```text
configs/config.yaml
```

Run:

```powershell
venv\Scripts\python.exe -m src.training.train_landsat
```

The current setup uses:

- Pix2Pix conditional GAN
- Dynamic U-Net generator (depth computed from input size)
- PatchGAN discriminator
- L1, adversarial, perceptual, SSIM, and chroma losses
- 500 epochs with linear LR decay from epoch 450
- Automatic mixed precision (AMP) training
- Checkpoint saving for best, latest, and final models
- Deterministic resume from any checkpoint

Checkpoints are written under:

```text
outputs/models/
```

Large checkpoint files are intentionally ignored by Git.

## Inference

The production inference engine is:

```text
src/inference/landsat_inference.py
```

It loads the trained checkpoint:

```text
outputs/models/best/pix2pix_landsat_best.pth
```

Run the standalone inference test:

```powershell
venv\Scripts\python.exe scripts\evaluation\test_inference.py
```

## Web App

InfraNova AI includes a full-stack web application for interactive colorization:

- **Backend**: FastAPI REST API (`api/main.py`) serving the PyTorch model
- **Frontend**: React app (`web/`) with a dark-themed UI built with Vite

### Running the Web App

**1. Start the backend:**

```powershell
venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**2. Start the frontend:**

```powershell
cd web
npm install
npm run dev
```

**3. Open http://localhost:5173** in your browser.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (returns model status) |
| `/colorize` | POST | Upload a thermal image, returns colorized RGB PNG |
| `/thermal-preview` | POST | Upload a thermal image, returns INFERNO heatmap preview |

### Features

- Drag-and-drop or click-to-upload thermal images
- Supports `.tif`, `.tiff`, `.png`, `.jpg`, `.npy` formats
- **Interactive Comparison Slider** to smoothly compare thermal vs RGB
- **CLAHE Post-Processing** toggle for enhanced contrast and saturation
- Optional test-time augmentation (TTA) toggle
- Real-time API health status indicator
- Inference time display
- Download colorized output as PNG

## Docker

### Development (Docker Compose)

For local development with hot-reloading, use Docker Compose which runs two separate containers:

- **`api`**: FastAPI backend serving the PyTorch model on port `8000`
- **`web`**: React frontend (Vite dev server) on port `5173`

```powershell
docker compose up --build
```

Both services mount the local source code for hot-reloading. Edit files locally and changes will be reflected automatically.

### Production (Multi-Stage Build)

For production, the root `Dockerfile` uses a multi-stage build that compiles the React frontend and serves everything from a single FastAPI container on port `8000`:

```powershell
docker build -t infranova-ai .
docker run -p 8000:8000 -v ./outputs:/app/outputs infranova-ai
```

The API container installs CPU PyTorch by default. To use a CUDA build, override the build arg:

```powershell
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t infranova-ai .
```

## CI/CD

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on every push and pull request to `main`:

1. **Lint**: Runs `ruff check` to catch code quality issues.
2. **Test**: Runs `pytest tests/` against the full test suite.

The pipeline uses Python 3.11 with CPU-only PyTorch to keep CI fast.

## Verification Commands

Useful quick checks:

```powershell
venv\Scripts\python.exe -m compileall -q src demo scripts
venv\Scripts\python.exe -m pip check
venv\Scripts\python.exe -m pytest tests/
```

## Notes

- `data/`, `outputs/`, `logs/`, and generated outputs are intentionally not committed.
- The model output is best described as RGB-like visual synthesis, not exact color recovery.
- For trustworthy reported metrics, evaluate on region-level splits rather than random patch-level splits.
- For a complete understanding of the system, start with `docs/project/MASTER_CODEBASE_GUIDE.md`.
