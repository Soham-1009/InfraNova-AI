# InfraNova AI — Execution Flow

This document traces the exact execution flow of the major pipelines in the project. It explicitly links each step to the file and function responsible.

## A. Dataset Acquisition Flow

How data moves from Google Earth Engine to raw NumPy arrays.

**START**
→ `scripts/download/download_landsat9.py` (`main()`)
→ `init_gee()` (Authenticates with Google Earth Engine)
→ `get_landsat9_collection()` (Filters L9 imagery by date and bounds)
→ `download_patch()` (Downloads specific regions using `geemap`)
→ **output**: Raw GeoTIFF files for ST_B10 (Thermal) and SR_B2/3/4 (RGB)
→ **next**: Preprocessing Flow

## B. Preprocessing Flow

How raw GeoTIFFs become aligned 128×128 patches.

**START**
→ `scripts/preprocessing/process_landsat_patches.py` (`main()`)
→ `process_scene()`
→ Extracts overlapping crops of 128×128 from the 100m resampled thermal and RGB GeoTIFFs
→ `save_patch()`
→ **output**: Sample directories containing `tir_100m.npy` and `rgb_100m.npy`
→ **next**: Dataset Validation Flow

## C. Dataset Validation Flow

How bad patches (clouds, empty space) are removed.

**START**
→ `scripts/preprocessing/validate_patch_dataset.py` (`main()`)
→ Scans generated `.npy` patches
→ Identifies patches with all-zeros, NaNs, or extremely low variance
→ `scripts/preprocessing/quarantine_small.py`
→ Moves invalid patches to a `quarantine/` directory
→ **output**: Clean dataset
→ **next**: Train Flow

## D. Training Flow

The end-to-end training loop for Pix2Pix.

**START**
→ `src/training/train_landsat.py` (`main()`)
→ `load_config()` (Loads `configs/config.yaml`)
→ `build_dataloaders()` 
  → Instantiates `src/datasets/landsat9_dataset.py` (`Landsat9Dataset`)
  → Wraps in `LandsatBatchAdapter` to provide `{"ir", "rgb"}` dictionary
→ `Pix2Pix()` Initialization (`src/models/pix2pix/pix2pix.py`)
  → Loads `GeneratorUNetDynamic` and `PatchDiscriminator`
→ `Trainer()` Initialization (`src/training/trainer.py`)
  → Sets up Adam optimizers, GradScaler (AMP), and `CombinedLoss`
→ **Epoch Loop** (`trainer.train_one_epoch()`)
  → **Discriminator Step**:
    → `model.generate(ir)` (Creates fake RGB)
    → `trainer._disc_loss_multi_scale()` (Computes D loss on real and fake)
    → `scaler.scale(d_loss).backward()`
    → `scaler.step(optimizer_d)`
  → **Generator Step**:
    → `model.generate(ir)`
    → `trainer._disc_forward(ir, fake_rgb)` (Gets D's prediction on fake)
    → `CombinedLoss.forward()` (Computes GAN + L1 + Perceptual + SSIM + Chroma loss)
    → `scaler.scale(g_loss).backward()`
    → `scaler.step(optimizer_g)`
→ `trainer.validate()`
  → Computes PSNR, SSIM, Color Histogram Distance, Lab Error
→ `ModelCheckpoint.step()` (`src/training/callbacks.py`)
  → Saves `best.pth` and `latest.pth` based on validation SSIM
→ **output**: Trained `.pth` models, `logs/training.csv`, visualizations
→ **next**: Web Inference Flow

## E. Web API Inference Flow

How a user's uploaded image becomes a colorized output.

**START**
→ User Drag & Drop in Browser
→ React POST request to `http://localhost:8000/colorize`
→ `api/main.py` (`colorize()`)
→ `get_engine()` (Lazy-loads `InferenceEngine` from `demo/inference.py`)
→ `engine.predict(image, use_tta=True)`
  → `demo/utils.py` (`preprocess_ir_image()`)
    → Converts to single-band, applies percentile stretch (p1 to p99), resizes to 128×128, scales to `[-1, 1]`
    → **Shape**: `[1, 1, 128, 128]`
  → `model.generate(tensor)` (Runs through Dynamic U-Net)
    → **Shape**: `[1, 3, 128, 128]`
  → TTA (Test-Time Augmentation):
    → Repeats inference for horizontal flip, vertical flip, and 180° rotation
    → Inverts transforms on outputs and averages the 4 tensors
  → `demo/utils.py` (`postprocess_output()`)
    → Scales from `[-1, 1]` to `[0, 255]`
    → Converts tensor to PIL RGB Image
→ Fast API encodes PIL Image as PNG buffer
→ `StreamingResponse` returned to React
→ **output**: Colorized image rendered in browser
