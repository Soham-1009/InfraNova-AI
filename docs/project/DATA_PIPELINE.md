# InfraNova AI — Data Pipeline

This document explains the complete dataset lifecycle from raw satellite imagery to PyTorch tensors ready for the Pix2Pix GAN.

## 1. Source Data

- **Satellite**: Landsat 9
- **Source**: Google Earth Engine (Level-2 Surface Reflectance and Surface Temperature)
- **Inputs used**:
  - `ST_B10`: Thermal Infrared (TIR) - Native resolution 100m. Contains surface temperature estimates.
  - `SR_B4`: Red (Visible) - Native resolution 30m.
  - `SR_B3`: Green (Visible) - Native resolution 30m.
  - `SR_B2`: Blue (Visible) - Native resolution 30m.

## 2. Preprocessing & Patch Creation

### Resolution Alignment
Because Landsat 9 captures thermal data at 100m/px and RGB data at 30m/px, they must be aligned. The preprocessing pipeline resamples the RGB bands to 100m/px to match the thermal resolution, creating a 1:1 spatial mapping.

### Patch Extraction
The continuous satellite swaths are cropped into small, fixed-size patches suitable for CNN training.
- **Dimensions**: 128×128 pixels (12.8 km × 12.8 km footprint).
- **Format**: Saved as raw NumPy `.npy` arrays to preserve floating-point precision (avoiding JPEG/PNG compression artifacts or 8-bit truncation).

For each spatial crop, the pipeline generates a sample folder (e.g., `data/landsat9/splits/train/patch_0001/`) containing:
- `tir_100m.npy` (shape: `[128, 128]`)
- `rgb_100m.npy` (shape: `[3, 128, 128]`)
- *Note: `tir_200m.npy` (64×64) is also generated for future Super-Resolution tasks but is not currently used for colorization.*

## 3. Dataset Auditing & Quarantine

Satellite imagery often contains invalid data (clouds, sensor edges, ocean). The `validate_patch_dataset.py` script filters these:
- **NoData handling**: Identifies and flags patches containing significant `NaN` values or zeroes.
- **Variance filtering**: Removes completely uniform patches (e.g., solid black or solid white).
- **Quarantine**: Bad patches are moved to a `quarantine/` folder so they don't pollute the training set.

## 4. PyTorch Dataset (`Landsat9Dataset`)

During training, `src/datasets/landsat9_dataset.py` loads the `.npy` files and prepares them for the model.

### 4.1 Normalization
Pix2Pix requires inputs and targets in the range `[-1, 1]` because the Generator's final activation is a `Tanh`.

- **Local Normalization (Default)**: For each patch, the 2nd (p2) and 98th (p98) percentiles are calculated. Pixel values are clipped to `[p2, p98]` and then linearly mapped to `[-1, 1]`. This "percentile stretching" ensures high contrast regardless of the absolute temperature of the region.
- **Global Normalization (Available)**: Uses precomputed p2 and p98 statistics across the entire dataset. (Currently disabled in config).

### 4.2 Data Augmentation
To prevent overfitting, spatial and radiometric augmentations are applied on-the-fly to the training set:
- **Spatial (Paired)**: 
  - Horizontal flip (p=0.5)
  - Vertical flip (p=0.5)
  - Random 90° rotation
- **Radiometric (Paired)**:
  - Brightness shift ±10% (p=0.3)
  - Contrast adjustment ±10% (p=0.3)
- **Noise (Input only)**:
  - Gaussian noise on the TIR input (p=0.3, σ=0.02) to simulate sensor noise and make the generator robust.

### 4.3 Output Format
The `__getitem__` method yields a dictionary:
```python
{
    'ir': torch.Tensor,   # Shape: [1, 128, 128], dtype: float32, range: [-1, 1]
    'rgb': torch.Tensor,  # Shape: [3, 128, 128], dtype: float32, range: [-1, 1]
    'name': str           # e.g., 'patch_0001'
}
```
*Note: The `LandsatBatchAdapter` in `train_landsat.py` ensures compatibility with the Trainer.*

## 5. Splits

The dataset uses a **region-based split**. Patches from the same geographic region are grouped strictly into either train, val, or test sets. This prevents spatial data leakage (where the model memorizes adjacent overlapping patches rather than learning to generalize).

## 6. Kaggle Optimization (`splits.zip`)

The raw Landsat 9 extraction can take upwards of 15GB of disk space (including `raw/`, `patches/`, and `discarded/`). However, the PyTorch dataset **only requires the `splits/` directory**. 

To significantly speed up Kaggle dataset updates, we compress only the `data/landsat9/splits/` directory into a 4.6GB `splits.zip` file. This is the only file pushed to Kaggle for training.
