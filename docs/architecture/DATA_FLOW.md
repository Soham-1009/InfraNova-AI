# Data Flow Specification — InfraNova AI

**Document Version:** 2.0  
**Date:** 2026-08-26  
**Auditor:** ML Engineer & Data Architect  

---

## 1. End-to-End Data Pipeline Flow

```
+-----------------------------------------------------------------------------------+
| 1. RAW INGESTION (Google Earth Engine)                                            |
|    - Landsat 9 Collection 2 Level 2 Surface Reflectance (SR_B2, SR_B3, SR_B4)     |
|    - Landsat 9 Collection 2 Level 2 Surface Temperature (ST_B10, ST_B11)          |
|    - Format: Multi-band GeoTIFFs organized into distinct geographic region folders|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. PATCH EXTRACTION & FILTERING (scripts/preprocessing/process_landsat_patches.py)|
|    - Resamples to 100m grid (128×128) using cv2.INTER_AREA                        |
|    - Sliding window extraction with 75% overlap (stride = 32 pixels at 100m)      |
|    - Validity Filter: skips patches with >50% zero/nodata or any NaN/Inf          |
|    - Writes per-sample files:                                                     |
|        tir_100m.npy     (128, 128, float32)   - Band 10 thermal                   |
|        tir_b11_100m.npy (128, 128, float32)   - Band 11 thermal                   |
|        rgb_100m.npy     (3, 128, 128, float32)- True color RGB (Red, Green, Blue) |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. GEOGRAPHIC SPLITTING (scripts/preprocessing/split_patches.py)                  |
|    - Groups samples by parent geographic region ID                                |
|    - Deterministic partition: 80% train / 10% val / 10% test (Seed = 42)          |
|    - Zero geographic spatial leakage across splits                                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. PYTORCH DATA LOADER (src/datasets/landsat9_dataset.py)                         |
|    - Local Normalization:                                                         |
|        B10: p2/p98 stretch to [-1, 1]                                             |
|        B11: p2/p98 stretch to [-1, 1] (independent from B10)                      |
|        RGB: per-channel p2/p98 stretch to [-1, 1]                                 |
|    - Stacks input channels: [B, 2, 128, 128]                                      |
|    - Paired Data Augmentations (train only):                                      |
|        H-flip, V-flip, 90° rotations, brightness/contrast jitter, Gaussian noise  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. MODEL FORWARD PASS (Dual Generator Architecture)                                |
|    - Input: [B, 2, 128, 128] (B10 and B11 dual-band thermal radiance)             |
|    - Primary Production Generator: Pix2PixHDGlobalResNetGenerator (11.37M params) |
|        * Single-scale 9 ResNet bottleneck blocks at 256 channels -> [B, 3, 128, 128] |
|        * 46.83% parameter reduction vs legacy baseline, zero high-frequency blur  |
|    - Legacy Reference Baseline: Pix2PixHDGenerator (21.38M parameters)             |
|        * Global Coarse Branch: [B, 2, 64, 64] -> Global Latent [B, 128, 64, 64]    |
|        * Local Fine Branch:   [B, 2, 128, 128] + Global Latent -> [B, 3, 128, 128] |
|    - Output Activation: Tanh scaled to [-1, 1]                                     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 6. POSTPROCESSING & INFERENCE (src/inference/landsat_inference.py & API)          |
|    - Strict Loading: load_pix2pix_model() validates architecture & shapes strictly|
|    - Denormalization: (tensor + 1.0) / 2.0 -> [0, 1] * 255.0 -> uint8 RGB [H,W,3] |
|    - Optional TTA: Averaging across 4 geometric transforms                        |
|    - Optional CLAHE: LAB color space lightness enhancement (tile = 8, clip = 2.0) |
|    - Full-Raster Tiling: 128x128 sliding window with 2D Cosine boundary blending  |
+-----------------------------------------------------------------------------------+
```