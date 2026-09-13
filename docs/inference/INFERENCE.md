# Inference Pipeline Documentation — InfraNova AI

**Document Version:** 2.0  
**Date:** 2026-08-26  
**Modules**: `src/inference/landsat_inference.py` & `demo/inference.py`  

---

## 1. Inference Engines

InfraNova AI provides two complementary inference engines:

### 1. `LandsatColorizationInference` (`src/inference/landsat_inference.py`)
- **Use Case**: Production CLI and GIS pipeline batch inference.
- **Input Formats**: PIL Images, 2D/3D NumPy arrays, single-band and multi-band TIFFs.
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
  - Automatic architecture detection: examines checkpoint state dictionary (`res_blocks` vs `downsample_local`) to instantiate either `Pix2PixHDGlobalResNetGenerator` (Exp9 candidate, 11.37M params) or `Pix2PixHDGenerator` (Production, 21.38M params).
  - TTA ensemble inference with automated denormalization and RGB clipping to $[0, 255]$.

---

## 2. Supported Checkpoints

| Checkpoint Identifier | Path | Architecture | Status | SHA-256 |
| :--- | :--- | :--- | :---: | :--- |
| **Production Baseline** | `outputs/best/pix2pix_landsat_best.pth` | `Pix2PixHDGenerator` (21.38M) | **Active Production** | `4604d36d07a0fb4c0696a53040c17004084068c51cc745a23f69b76c8baf6aa8` |
| **Exp9 Candidate** | `outputs/exp9/best/pix2pix_landsat_best.pth` | `Pix2PixHDGlobalResNetGenerator` (11.37M) | **Audited Candidate** | `71bbda3f31b85e7e741b26d5ce0ff398a0394c7dd4ef6ff5452f31f7e0400382` |

---

## 3. Running Inference from CLI

```powershell
# Run inference using Production Checkpoint (Pix2PixHD Dual-Scale)
python demo/inference.py --checkpoint outputs/best/pix2pix_landsat_best.pth --generator_type hd

# Run inference using Exp9 Candidate Checkpoint (Global ResNet)
python demo/inference.py --checkpoint outputs/exp9/best/pix2pix_landsat_best.pth --generator_type resnet
```
