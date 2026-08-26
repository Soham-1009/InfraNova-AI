# System Architecture Specification — InfraNova AI

**Document Version:** 2.0  
**Date:** 2026-08-26  
**Auditor:** Systems Architect & Senior ML Engineer  

---

## 1. High-Level System Architecture

InfraNova AI is structured into four decoupled subsystems:
1. **Data Ingestion & Preprocessing Subsystem**: Downloads Earth Engine imagery, resamples, filters invalid/blank pixels, and generates geographic spatial splits.
2. **Model Training & Optimization Subsystem**: Multi-loss optimization on dual GPUs using `DataParallel`, linear learning rate decay, and memory-safe checkpointing.
3. **Inference & Serving Subsystem**: Standalone production inference engine with TTA, confidence estimation, and 3-band GIS GeoTIFF export.
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
|                                             |                                      |
|                                             v                                      |
|                                    outputs/final/ (checkpoints & logs)             |
+------------------------------------------------------------------------------------+
                                              |
                                              v
+------------------------------------------------------------------------------------+
|                               SERVING & APPLICATION SUBSYSTEM                      |
|                                                                                    |
|                       FastAPI REST Backend (api/main.py :8000)                     |
|                         ├── POST /colorize                                         |
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

### 2.1. Checkpoint Compatibility
- When training under `nn.DataParallel`, PyTorch prefixes all module keys with `.module.`.
- `src/utils/checkpoint.py`, `demo/inference.py`, and `src/inference/landsat_inference.py` strip this prefix dynamically upon loading:
  ```python
  clean_state_dict = {k.replace(".module.", "."): v for k, v in state_dict.items()}
  ```

### 2.2. Production Containerization (`Dockerfile`)
- **Stage 1 (`frontend-build`)**: Installs Node.js dependencies and executes `npm run build` to generate optimized production assets in `/web/dist`.
- **Stage 2 (`python:3.11-slim`)**: Installs PyTorch, OpenCV system libraries (`libgl1`, `libglib2.0-0`), and Python dependencies; copies `/web/dist` to `/app/web/dist` and serves both the API and frontend on port 8000.