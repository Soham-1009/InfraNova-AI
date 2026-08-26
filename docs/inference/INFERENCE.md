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
- **Use Case**: REST API backend singleton (`api/main.py`).
- **Features**: Fast in-memory single-image and batch prediction supporting both PIL and NumPy inputs.

---

## 2. Running Inference from CLI

```powershell
# Run inference on a thermal image patch using the final 250-epoch checkpoint
python scripts/evaluation/test_inference.py --checkpoint outputs/final/epoch_250.pth --input data/sample_thermal.tif --output outputs/colorized_output.png
```
