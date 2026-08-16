# InfraNova AI — API and Web App

This document outlines the architecture of the modern web stack (React + FastAPI) that replaced the original Streamlit demo.

## 1. Backend (FastAPI)

The backend exposes the ML inference engine over a RESTful API.

- **Location**: `api/main.py`
- **Port**: 8000 (by default via uvicorn)

### Endpoints

#### `GET /health`
Returns the status of the API and checks if the Pix2Pix model has been loaded into memory yet. (The model lazy-loads on the first inference request to speed up boot time).

#### `POST /colorize`
- **Accepts**: Multipart form-data containing a `file` (`.tif`, `.png`, `.jpg`, `.npy`). Optional query param `tta=true/false`.
- **Action**:
  1. Parses image bytes into PIL Image or NumPy array.
  2. Passes to `InferenceEngine.predict(use_tta=tta)`.
  3. Encodes resulting PIL RGB Image to PNG.
- **Returns**: Binary PNG stream. `X-Inference-Time` and `X-Model` are attached as response headers.

#### `POST /thermal-preview`
- **Accepts**: Same as `/colorize`.
- **Action**: Generates a pseudo-color map (using OpenCV's `INFERNO` colormap) representing the raw thermal data, applying identical percentile stretching to the inference engine.
- **Returns**: Binary PNG stream.

## 2. Frontend (React)

The frontend provides a polished, interactive UI for comparing thermal inputs with generated RGB outputs.

- **Location**: `web/`
- **Framework**: React 19 + Vite 6
- **Styling**: Custom CSS (`index.css`), utilizing CSS variables for a dark theme and "glassmorphism" effects.
- **Port**: 5173 (by default via `npm run dev`)

### Core Components (`web/src/App.jsx`)

1. **Drag-and-Drop Zone**: Allows users to select or drop local image files.
2. **Side-by-Side Comparison**: Uses a custom slider implementation overlaid on two absolutely positioned images. Allows the user to swipe left/right to reveal the RGB generation over the Thermal preview.
3. **API Integration**: Sends two asynchronous POST requests (`/colorize` and `/thermal-preview`) in parallel to fetch both visualization layers.
4. **State Management**: Handles Loading (spinner), Error (toast notifications), and Result states.

### Key CSS Features (`web/src/index.css`)
- Extensively uses CSS variables (`--bg`, `--surface`, `--accent`, etc.) for consistent theming.
- The `before/after` slider uses CSS `clip-path` to dynamically crop the top image based on mouse position.
- Minimal dependencies (no Tailwind, pure CSS) per project guidelines.

## 3. Running the Stack

To run the full stack locally for development:

**Terminal 1 (Backend):**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd web
npm run dev
```

*Note: The React app is configured to point to `http://localhost:8000` for API requests.*
