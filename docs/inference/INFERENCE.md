# Inference Pipeline Documentation — InfraNova AI

**Document Version:** 2.1
**Date:** 2026-09-13
**Modules**: `src/inference/landsat_inference.py` & `demo/inference.py`  

---

## 1. Inference Engines

InfraNova AI provides two complementary inference engines:

### 1. `LandsatColorizationInference` (`src/inference/landsat_inference.py`)
- **Use Case**: Production CLI and GIS pipeline batch inference.
- **Input Formats**: PIL images, 2D/3D NumPy arrays, and paired Band 10/Band 11 arrays.
- **Production Input**: The active model expects two thermal channels (Band 10 and Band 11). A single-band input is duplicated only for backwards-compatible interactive use; it is not a substitute for paired thermal data.
- **Features**:
  - Robust 2nd/98th percentile stretching with flat-image protection.
  - Test-Time Augmentation (TTA) with 4 geometric transforms (Identity, H-flip, V-flip, 180° rotation).
  - Confidence scoring derived from inter-augmentation variance:
    $$\text{Confidence} = \exp(-8.0 \cdot \bar{\sigma}_{\text{TTA}})$$
  - 3-band GIS GeoTIFF export (saving in B, G, R layer order for standard GIS visualization).

### 2. `InferenceEngine` (`demo/inference.py`)
- **Use Case**: REST API backend singleton (`api/main.py`) and interactive demonstrations.
- **Features**:
  - Fast in-memory single-image and batch prediction supporting both PIL and NumPy inputs.
  - Strict checkpoint loading: validates architecture metadata and tensor shapes before constructing the generator. The active production checkpoint uses `Pix2PixHDGlobalResNetGenerator` (Exp9, 11.37M parameters).
  - TTA ensemble inference with automated denormalization and RGB clipping to $[0, 255]$.

---

## 2. Supported Checkpoints

| Checkpoint Identifier | Path | Architecture | Status | SHA-256 |
| :--- | :--- | :--- | :---: | :--- |
| **Active Production (Exp9)** | `outputs/best/pix2pix_landsat_best.pth` | `Pix2PixHDGlobalResNetGenerator` (11.37M) | **Active Production** | `71bbda3f31b85e7e741b26d5ce0ff398a0394c7dd4ef6ff5452f31f7e0400382` |
| **Historical Exp9 Source Copy** | `outputs/exp9/best/pix2pix_landsat_best.pth` | `Pix2PixHDGlobalResNetGenerator` (11.37M) | Historical copy | `71bbda3f31b85e7e741b26d5ce0ff398a0394c7dd4ef6ff5452f31f7e0400382` |

---

## 3. Running Batch Inference from CLI

```powershell
# Run the active production checkpoint over a thermal image or directory
python scripts/deployment/batch_inference.py --input-dir path/to/thermal --output-dir outputs/inference
```

`demo/inference.py` exposes the `InferenceEngine` class for the REST API and programmatic use; it is not a command-line entry point.
