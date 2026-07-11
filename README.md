# InfraNova AI

InfraNova AI converts Landsat 9 thermal infrared imagery into RGB-like satellite imagery using a Pix2Pix conditional GAN.

The project is designed around a simple idea: thermal images show heat patterns, while most humans and computer vision tools are easier to work with when imagery looks like normal RGB satellite data. The model learns from paired Landsat 9 thermal and RGB bands, then generates a plausible RGB view from a single thermal input.

## What This Project Does

- Downloads Landsat 9 thermal and RGB bands from Google Earth Engine.
- Organizes raw `.tif` exports into region folders.
- Builds paired training patches from thermal infrared and RGB bands.
- Trains a Pix2Pix GAN for TIR-to-RGB colorization.
- Runs inference from a checkpoint.
- Provides a Streamlit app for uploading thermal images and viewing generated RGB output.

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
Patch preparation
    |
    v
Pix2Pix model
    |
    |-- U-Net generator       -> creates RGB image
    |-- PatchGAN discriminator -> checks realism
    |
    v
RGB-like satellite output
```

## Repository Layout

```text
configs/
  config.yaml                 Training and inference configuration

src/
  datasets/                   Landsat 9 dataset loader
  models/pix2pix/             Generator, discriminator, Pix2Pix wrapper
  training/                   Losses, trainer, callbacks, scheduler
  inference/                  Production inference engine
  utils/                      Checkpoint and logging helpers

demo/
  streamlit_app.py            Web demo
  inference.py                Demo inference wrapper
  utils.py                    Demo preprocessing and display helpers

docs/
  architecture.md             Model architecture notes
  methodology.md              Thermal-to-RGB methodology
  training_strategy.md        Training plan and monitoring notes

download_landsat9.py          Earth Engine export script
organize_files.py             Organizes downloaded .tif exports
process_landsat_patches.py    Builds .npy patch samples
split_patches.py              Creates region-level train/val/test splits
requirements.txt              Main Python dependencies
```

## Setup

Use the existing virtual environment if it is already present:

```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

Or run commands directly through the environment:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### PyTorch Install Options

`requirements.txt` keeps the PyTorch packages listed, but you should install the build that matches your machine before the rest of the dependencies.

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

## Data Pipeline

### 1. Export Landsat 9 data

```powershell
venv\Scripts\python.exe download_landsat9.py
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
venv\Scripts\python.exe organize_files.py --source path\to\downloaded\folder
```

This creates per-region folders under:

```text
data/landsat9/input/
```

### 3. Build patches

```powershell
venv\Scripts\python.exe process_landsat_patches.py
```

This creates samples under:

```text
data/landsat9/patches/
```

Each sample contains:

- `tir_200m.npy`
- `tir_100m.npy`
- `rgb_100m.npy`

### 4. Create splits

```powershell
venv\Scripts\python.exe split_patches.py --overwrite
```

The split is region-level. Patches from the same region are kept in only one of train, validation, or test to avoid evaluation leakage. The splitter also keeps small datasets usable by avoiding empty train/val partitions when there are only a few regions.

## Training

Training is configured in:

```text
configs/config.yaml
```

Run:

```powershell
venv\Scripts\python.exe src\training\train_landsat.py
```

The current setup uses:

- Pix2Pix conditional GAN
- U-Net generator
- PatchGAN discriminator
- L1, adversarial, perceptual, and SSIM losses
- 250 epochs
- linear learning-rate decay near the end of training
- checkpoint saving for best, latest, and final models

Checkpoints are written under:

```text
checkpoints/
```

Large checkpoint files are intentionally ignored by Git.

## Inference

The production inference engine is:

```text
src/inference/landsat_inference.py
```

It loads the trained checkpoint:

```text
checkpoints/best/pix2pix_landsat_best.pth
```

The existing checkpoint can be used for demo and inference. Retraining is only needed if you want a clean benchmark using the latest region-level split and corrected training settings.

## Streamlit Demo

Run:

```powershell
venv\Scripts\python.exe -m streamlit run demo\streamlit_app.py
```

The app lets you upload a thermal image and generate an RGB-like output.
It includes:

- a light/dark mode toggle
- optional test-time augmentation
- optional contrast enhancement
- download of the generated RGB image

## Docker

The repo now includes a Docker setup for the Streamlit demo:

```powershell
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

The container mounts the repo into `/app`, so you can edit files locally and refresh the browser without rebuilding every time.

The Docker image installs CPU PyTorch explicitly before the rest of the requirements, which keeps the demo image smaller and avoids pulling CUDA-only wheels unless you choose to change the base image yourself.

You can also run one-off scripts inside the container, for example:

```powershell
docker compose run --rm app python process_landsat_patches.py
docker compose run --rm app python src/training/train_landsat.py
```

For GPU training on Docker, swap the base image to a CUDA-enabled PyTorch image and run the container with NVIDIA runtime support. The current Dockerfile is aimed at the demo and CPU-friendly utility scripts.

## Working Locally

If you change only Python source files while the Docker volume mount is active, restart the container or refresh the browser.

If you change `Dockerfile`, `requirements.txt`, or system-level dependencies, rebuild the image:

```powershell
docker compose up --build
```

## Verification Commands

Useful quick checks:

```powershell
venv\Scripts\python.exe -m compileall -q download_landsat9.py organize_files.py process_landsat_patches.py split_patches.py src demo
venv\Scripts\python.exe -m pip check
```

## Notes

- `data/`, `checkpoints/`, `logs/`, and generated outputs are intentionally not committed.
- The model output is best described as RGB-like visual synthesis, not exact color recovery.
- For trustworthy reported metrics, evaluate on region-level splits rather than random patch-level splits.
