# Data Pipeline Documentation — InfraNova AI

**Document Version:** 2.0  
**Date:** 2026-08-26  
**Module**: `src/datasets/` & `scripts/preprocessing/`  

---

## 1. Landsat 9 Satellite Spectral Bands

| Band Identifier | Spectral Category | Wavelength Range | Nominal Ground Resolution | Usage in InfraNova AI |
| :--- | :--- | :---: | :---: | :--- |
| **SR_B2** | Visible Blue | $0.45\text{--}0.51\,\mu\text{m}$ | 30m | RGB Target (Blue Channel) |
| **SR_B3** | Visible Green | $0.53\text{--}0.59\,\mu\text{m}$ | 30m | RGB Target (Green Channel) |
| **SR_B4** | Visible Red | $0.64\text{--}0.67\,\mu\text{m}$ | 30m | RGB Target (Red Channel) |
| **ST_B10** | Thermal Infrared 1 (TIRS-2) | $10.60\text{--}11.19\,\mu\text{m}$ | 100m | **Primary Input (Channel 0)** |
| **ST_B11** | Thermal Infrared 2 (TIRS-2) | $11.50\text{--}12.51\,\mu\text{m}$ | 100m | **Secondary Input (Channel 1)** |

---

## 2. Preprocessing & Patch Generation

Run patch generation on raw downloaded scenes:

```powershell
python scripts/preprocessing/process_landsat_patches.py --input-dir data/raw --output-dir data/landsat9_b10_b11/patches
```

### Preprocessing Steps:
1. **Resampling**: Uses `cv2.INTER_AREA` to align 30m optical and 100m thermal rasters to exact $100\text{m}$ ($128\times 128$) and $200\text{m}$ ($64\times 64$) grids.
2. **Sliding Window Extraction**: Stride of 32 pixels at $100\text{m}$ resolution (75% spatial overlap).
3. **Data Quality & Blank Filtering**:
   - Discards any patch with $>50\%$ zero pixels.
   - Discards patches containing any non-finite values (NaN / Inf) in either thermal or optical bands.
4. **Sample Artifacts Saved**:
   - `tir_100m.npy`: Float32 array $(128, 128)$ for Band 10.
   - `tir_b11_100m.npy`: Float32 array $(128, 128)$ for Band 11.
   - `rgb_100m.npy`: Float32 array $(3, 128, 128)$ for RGB target.

---

## 3. Geographic Dataset Splitting

```powershell
python scripts/preprocessing/split_patches.py --input-dir data/landsat9_b10_b11/patches --output-dir data/landsat9_b10_b11/splits --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42
```

### Partitioning Guarantees:
- **Zero Spatial Leakage**: Patches from the same geographic region are strictly assigned to either `train/`, `val/`, or `test/`.
- **Reproducibility**: Deterministic shuffling using seed 42.

---

## 4. Normalization Modes (`Landsat9Dataset`)

1. **Local Percentile Normalization (Default)**:
   - Evaluates 2nd and 98th percentiles per patch.
   - Applied independently to Band 10 and Band 11 to preserve split-window thermal variance.
   - Normalizes to $[-1, 1]$.
2. **Global Dataset Normalization**:
   - Uses precomputed dataset statistics generated via `python scripts/preprocessing/compute_normalization_stats.py`.
