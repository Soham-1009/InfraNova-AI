# InfraNova AI — Known Bugs and Quirks

This document lists known issues in the codebase that developers should be aware of. 

## 1. High Priority / Breaking

### 1.1 Docker Configuration is Broken
- **Location**: `Dockerfile`, `docker-compose.yml`
- **Description**: The Dockerfiles run `CMD ["streamlit", "run", "demo/streamlit_app.py", ...]`. However, `streamlit_app.py` was deleted when the project migrated to React+FastAPI. 
- **Impact**: Any attempt to build and run the Docker container will immediately crash.
- **Fix Required**: Rewrite Dockerfile for a FastAPI/React deployment.

## 2. Medium Priority / Misleading

### 2.1 `src/losses/` Directory is Empty
- **Location**: `src/losses/`
- **Description**: The directory exists but contains no code.
- **Impact**: Misleading to new developers looking for the loss functions.
- **Fix Required**: Delete the directory. All losses are properly implemented in `src/training/losses.py`.

### 2.2 Inference Engine Image Size Validation
- **Location**: `src/inference/landsat_inference.py` (Line 58)
- **Description**: The `LandsatColorizationInference` class throws an error if `image_size` is not a multiple of 256. However, the model is trained at 128x128.
- **Impact**: If this class is used with the current checkpoints, it will reject the valid 128x128 inputs.
- **Workaround**: The production API correctly uses `demo/inference.py` instead, which validates for multiples of 128.

### 2.3 Checkpoint Architecture Metadata Hardcodes Image Size
- **Location**: `src/utils/checkpoint.py` (Line 96)
- **Description**: When saving a checkpoint, the `arch_info` metadata hardcodes `"image_size": 256` regardless of the actual config.
- **Impact**: If a future update relies on `arch_info` to automatically resize inputs, it will mistakenly resize to 256.

### 2.4 README Mentions `config_smoke.yaml`
- **Location**: `README.md`
- **Description**: The README references a `config_smoke.yaml` file for testing.
- **Impact**: The file does not exist in the `configs/` directory.
- **Workaround**: Use CLI overrides instead (`--overrides epochs=1 batch_size=4`).

## 3. Low Priority / Edge Cases

### 3.1 Model Desaturation
- **Location**: Model weights
- **Description**: The model frequently outputs images with low color saturation (brownish/grayish tints) compared to real RGB ground truth.
- **Impact**: Visual quality issue.
- **Workaround**: Increase `lambda_chroma` in config or implement CLAHE postprocessing in the UI.

### 3.2 Dataset Pipeline Strict Shape Validation
- **Location**: `src/datasets/landsat9_dataset.py` (Line 250)
- **Description**: The dataset throws a hard error if any patch on disk does not perfectly match the configured `image_size`. It does not attempt to resize them dynamically.
- **Impact**: Changing `image_size` in config from 128 to 256 will crash the dataloader until the entire dataset is regenerated from Earth Engine.
