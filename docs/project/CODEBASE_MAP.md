# InfraNova AI — Codebase Map

This document provides a detailed map of the entire repository. Use it to understand where things live, what they do, and the risk of modifying them.

## Directory Structure

- `api/`: FastAPI backend for production serving.
- `configs/`: YAML configuration files.
- `data/`: Dataset storage (ignored in git).
- `demo/`: Inference utilities and Web App backend hooks.
- `docs/`: Technical documentation (including this file).
- `experiments/`: (Empty) Intended for experiment tracking.
- `logs/`: Training metrics, CSVs, and tensorboard logs.
- `notebooks/`: Kaggle training environment definitions.
- `outputs/`: Saved models, generated visuals, and checkpoints.
- `scripts/`: Operational scripts (download, preprocessing, evaluation).
- `src/`: Core machine learning source code.
- `tests/`: Unit and integration test suite.
- `web/`: React frontend application.

## Core Modules (High Risk)

### `src/models/pix2pix/pix2pix.py`
- **PURPOSE**: Master wrapper for the Generator and Discriminator.
- **RESPONSIBILITY**: Exposes `generate()`, `discriminate()`, and handles device placement.
- **USED BY**: `src/training/train_landsat.py`, `demo/inference.py`
- **MODIFICATION RISK**: **HIGH**. Any changes to input/output shapes or device handling will break both training and inference.

### `src/models/pix2pix/generator_dynamic.py`
- **PURPOSE**: The production Generator architecture.
- **RESPONSIBILITY**: Translates 1-channel TIR tensors into 3-channel RGB tensors via a U-Net.
- **USED BY**: `pix2pix.py` (when `generator.implementation: dynamic`)
- **MODIFICATION RISK**: **HIGH**. Changes to channel counts, padding, or strides will invalidate all existing checkpoints.

### `src/models/pix2pix/discriminator.py`
- **PURPOSE**: The Discriminator architecture.
- **RESPONSIBILITY**: Judges the realism of concatenated TIR+RGB patch pairs.
- **USED BY**: `pix2pix.py`
- **MODIFICATION RISK**: **HIGH**. Contains both single-scale and multi-scale variants. Critical for adversarial stability.

### `src/training/trainer.py`
- **PURPOSE**: The main training loop.
- **RESPONSIBILITY**: Orchestrates forward/backward passes, metric calculation, logging, and checkpoint saving.
- **USED BY**: `src/training/train_landsat.py`
- **MODIFICATION RISK**: **HIGH**. Complex interactions with AMP (`GradScaler`), optimizer stepping, and multi-GPU `DataParallel`.

### `src/training/losses.py`
- **PURPOSE**: Defines all objective functions.
- **RESPONSIBILITY**: Implements GAN, L1, Perceptual, SSIM, and Chroma losses. Also contains color evaluation metrics.
- **USED BY**: `src/training/trainer.py`
- **MODIFICATION RISK**: **HIGH**. Loss scaling and target creation (especially for BCE GAN) are highly sensitive.

## Operational Modules (Medium Risk)

### `src/inference/landsat_inference.py`
- **PURPOSE**: Production API inference engine.
- **RESPONSIBILITY**: Loads models, applies percentile stretching, handles batching and TTA, and saves TIFFs.
- **USED BY**: Conceptually meant for backend, but currently superseded by `demo/inference.py` in the API.
- **MODIFICATION RISK**: **MEDIUM**. Requires careful handling of float32 to uint8 conversions.

### `demo/inference.py`
- **PURPOSE**: Web-app inference engine.
- **RESPONSIBILITY**: Connects the FastAPI backend to the Pix2Pix model. Handles PIL image conversions.
- **USED BY**: `api/main.py`
- **MODIFICATION RISK**: **MEDIUM**. Safely sandboxed from training, but powers the UI.

### `src/datasets/landsat9_dataset.py`
- **PURPOSE**: PyTorch Dataset definition.
- **RESPONSIBILITY**: Loads `.npy` arrays, applies normalization, applies data augmentations, and returns dicts.
- **USED BY**: `src/training/train_landsat.py`
- **MODIFICATION RISK**: **MEDIUM**. Changes to the augmentation pipeline affect training behavior, but not system stability.

## Web Modules (Low Risk)

### `api/main.py`
- **PURPOSE**: FastAPI backend.
- **RESPONSIBILITY**: Defines REST endpoints (`/colorize`, `/health`). Manages file uploads and CORS.
- **USED BY**: `web/src/App.jsx`
- **MODIFICATION RISK**: **LOW**. Standard web API logic.

### `web/src/App.jsx`
- **PURPOSE**: React frontend.
- **RESPONSIBILITY**: Provides user interface, image upload zone, and side-by-side comparison slider.
- **USED BY**: End user browser.
- **MODIFICATION RISK**: **LOW**. Purely visual.

## Legacy / Unused Files

- `src/models/pix2pix/generator.py`: Legacy static U-Net. Only works on 256×256 inputs. Kept for backward compatibility.
- `demo/streamlit_app.py` (Deleted, but referenced in Dockerfile): Replaced by React/FastAPI.
- `scripts/deployment/export/`: Contains ONNX/TorchScript export scripts that are currently untested in production.
