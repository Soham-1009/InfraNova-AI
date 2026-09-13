# Deployment & Serving Documentation — InfraNova AI

**Document Version:** 2.0  
**Date:** 2026-08-26  
**Modules**: `api/`, `web/`, `Dockerfile`, `docker-compose.yml`  

---

## 1. FastAPI REST Backend (`api/main.py`)

The REST API exposes the following endpoints:

| Endpoint | Method | Input Parameters | Output Response | Function |
| :--- | :---: | :--- | :--- | :--- |
| **`/health`** | `GET` | None | JSON `{"status": "ok", "model_loaded": bool}` | Liveness & model status probe |
| **`/colorize`** | `POST` | `file`: UploadFile (`.tif`, `.tiff`, `.png`, `.npy`), `tta`: bool | Streaming PNG (`image/png`) | Synthesizes true color RGB image |
| **`/thermal-preview`** | `POST` | `file`: UploadFile | Streaming PNG (`image/png`) | Renders thermal Inferno colormap |
| **`/postprocess/clahe`**| `POST`| `file`: UploadFile, `clip_limit`: float (2.0), `grid_size`: int (8) | Streaming PNG (`image/png`) | LAB lightness contrast enhancement |

---

## 2. React + Vite Interactive Frontend (`web/`)

- **Interactive Comparison Slider**: Side-by-side draggable split-view between input thermal radiance and synthesized RGB output.
- **Dynamic Post-Processing**: Real-time CLAHE contrast adjustment in LAB color space.
- **Test-Time Augmentation (TTA) Toggle**: Triggers multi-transform averaging for enhanced boundary clarity.
- **Export Controls**: Direct download of synthesized optical products as PNG or GIS-standard GeoTIFFs.

---

## 3. Containerized Deployment

### Multi-Stage Production Build (`Dockerfile`)
```powershell
# Build single production container containing FastAPI backend and React frontend
docker build -t infranova-ai .

# Run production container
docker run -p 8000:8000 -v ./outputs:/app/outputs infranova-ai
```

### Docker Compose Multi-Container Workflow (`docker-compose.yml`)
```powershell
# Start both backend (port 8000) and frontend development server (port 5173)
docker compose up --build
```

---

## 4. Model Serving Configuration & Checkpoint Governance

The inference service (`demo/inference.py` wrapped by `api/main.py`) supports dynamic architecture auto-detection:

- **Active Production Model (Promoted)**:
  - Path: `outputs/best/pix2pix_landsat_best.pth`
  - SHA-256: `71bbda3f31b85e7e741b26d5ce0ff398a0394c7dd4ef6ff5452f31f7e0400382`
  - Architecture: `Pix2PixHDGlobalResNetGenerator` (11,369,795 parameters)
  - Key Metrics: 13.745 dB PSNR, 0.4502 SSIM, 0.1723 MAE, 0.1968 rad SAM (11.28°), 0.3711 YOLOv8n F1@0.25
  - Known Trade-off: Histogram distance is 0.4607 vs legacy Production 0.3572
- **Archived Legacy Production Backup**:
  - Path: `outputs/best/pix2pix_landsat_backup_20260912_231938.pth`
  - SHA-256: `4604d36d07a0fb4c0696a53040c17004084068c51cc745a23f69b76c8baf6aa8`
  - Architecture: `Pix2PixHDGenerator` (21,383,238 parameters)
- **Rollback Metadata**: Maintained at `outputs/best/rollback_metadata.json` for one-command rollback capability if ever needed.
