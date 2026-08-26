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
