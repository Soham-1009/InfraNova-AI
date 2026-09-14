# System Architecture Specification — InfraNova AI

**Document Version:** 2.0  
**Date:** 2026-08-26  
**Auditor:** Systems Architect & Senior ML Engineer  

---

## 1. High-Level System Architecture

InfraNova AI is structured into four decoupled subsystems:
1. **Data Ingestion & Preprocessing Subsystem**: Downloads Earth Engine imagery, resamples, filters invalid/blank pixels, and generates geographic spatial splits.
2. **Model Training & Optimization Subsystem**: Multi-loss optimization on dual GPUs using `DataParallel`, linear learning rate decay, early stopping patience, and dual generator support (`hd` and `resnet`).
3. **Inference & Serving Subsystem**: Standalone production inference engine with TTA, confidence estimation, auto-detecting model architecture (`hd` or `resnet`), and 3-band GIS GeoTIFF export.
4. **Application & Web Subsystem**: FastAPI REST backend serving a containerized React + Vite frontend.

```
+------------------------------------------------------------------------------------+
|                                DATA & TRAINING SUBSYSTEM                           |
|                                                                                    |
|   Google Earth Engine                                                              |
|   (B2, B3, B4, B10, B11)                                                           |
|             |                                                                      |
|             v                                                                      |
|   process_landsat_patches.py -----> split_patches.py (80/10/10 by region)          |
|                                             |                                      |
|                                             v                                      |
|                                    Landsat9Dataset [B, 2, 128, 128]                |
|                                             |                                      |
|                                             v                                      |
|                                    Trainer (Dual T4 GPU DataParallel)              |
|                                    ├── GlobalResNetGenerator (Active Prod, 11.37M) |
|                                    └── Pix2PixHDGenerator (Legacy Baseline, 21.38M)|
|                                             |                                      |
|                                             v                                      |
|                                    outputs/ (checkpoints & logs)                   |
+------------------------------------------------------------------------------------+
                                              |
                                              v
+------------------------------------------------------------------------------------+
|                               SERVING & APPLICATION SUBSYSTEM                      |
|                                                                                    |
|                       FastAPI REST Backend (api/main.py :8000)                     |
|                         ├── POST /colorize (supports ?tta=true)                    |
|                         ├── POST /thermal-preview                                  |
|                         ├── POST /postprocess/clahe                                |
|                         └── GET  /health                                           |
|                                      ^                                             |
|                                      | HTTP REST Requests                          |
|                                      v                                             |
|                       React + Vite Frontend (web/src/App.jsx)                      |
|                         ├── Dual-Pane Comparison Slider                            |
|                         ├── CLAHE Dynamic Adjustment Slider                        |
|                         ├── TTA Quality Mode Toggle                                |
|                         └── GeoTIFF / PNG Image Downloader                         |
+------------------------------------------------------------------------------------+
```

---

## 2. Component Interoperability & Contracts

### 2.1. Checkpoint & Architecture Compatibility
- When training under `nn.DataParallel`, PyTorch prefixes module keys with `.module.`.
- `src/utils/checkpoint.py` provides the canonical `load_pix2pix_model()` utility, which strips `.module.` prefixes and infers model specifications (`resnet` vs `hd`, input channels, output channels, scales) directly from checkpoint weights and architecture metadata.
- **Strict Loading Guarantee**: All inference engines and evaluation scripts use strict shape validation (`strict=True`), rejecting incomplete or incompatible state dicts rather than silently leaving randomly initialized layers.
- **Thread Safety**: Serving engines (`api/main.py` and `demo/inference.py`) encapsulate checkpoint initialization with re-entrant thread locks (`threading.RLock`) to guarantee single-instance initialization under concurrent requests.

### 2.2. Production Containerization (`Dockerfile`)
- **Stage 1 (`frontend-build`)**: Installs Node.js dependencies and executes `npm run build` to generate optimized production assets in `/web/dist`.
- **Stage 2 (`python:3.11-slim`)**: Installs PyTorch, OpenCV system libraries (`libgl1`, `libglib2.0-0`), and Python dependencies; copies `/web/dist` to `/app/web/dist` and serves both the API and frontend on port 8000.