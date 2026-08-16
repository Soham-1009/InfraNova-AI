# InfraNova AI — Developer Guide

This document is for software engineers maintaining the web stack, API, deployment, and testing infrastructure of InfraNova AI.

## 1. Local Development Environment

### 1.1 Python Backend (API & ML)
We use standard `pip` for Python package management. Python 3.11+ is required.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1.2 Node Frontend (UI)
We use `npm` for the React application. Node 18+ is required.

```bash
cd web
npm install
```

## 2. Code Quality & Standards

The project uses `pyproject.toml` to enforce strict formatting and linting.

- **Formatter**: `black` (line length 120)
- **Linter**: `ruff` (rules: E, W, F, I, B, UP, SIM, RUF)
- **Import Sorter**: `isort` (profile: black)
- **Type Checker**: `mypy`

**Before committing any Python code, you MUST run:**
```bash
ruff check . --fix
black .
isort .
```

For the frontend, we use ESLint and Prettier (standard Vite setup).

## 3. Extending the API

If you need to add a new endpoint to the FastAPI backend:

1. Open `api/main.py`.
2. Define the new route (e.g., `@app.get("/stats")`).
3. If it requires ML inference, **do not** instantiate a new `InferenceEngine`. Use the singleton getter `get_engine()` to avoid crashing the server by loading 2x model weights into RAM.
4. Restart Uvicorn to test.

## 4. Modifying the UI

If you need to change the React UI:

1. All custom styling lives in `web/src/index.css`. We use CSS variables extensively for theming.
2. The core logic for the image slider is in `web/src/App.jsx`. If you adjust the layout, ensure the slider's `clip-path` math still accurately maps to the container boundaries.
3. Test responsive design: The app is fixed at `100vh`. Shrink your browser window vertically to ensure nothing overflows off the screen.

## 5. Working with Docker (Fixing the Build)

As noted in `BUGS.md`, the current Docker setup is broken. To fix it, you will need to:

1. Modify `Dockerfile` to create a multi-stage build.
2. In Stage 1: Build the React app (`npm run build`).
3. In Stage 2: Set up the Python environment, install `requirements.txt`, and run FastAPI (`uvicorn`).
4. Serve the React static files from FastAPI via `fastapi.staticfiles.StaticFiles(directory="web/dist", html=True)`.
5. Update `docker-compose.yml` to reflect these changes.

## 6. Testing

Run the test suite before submitting a PR:
```bash
pytest tests/ -v
```

If you add a new loss function or model architecture, you **must** add a corresponding unit test to `tests/`.
