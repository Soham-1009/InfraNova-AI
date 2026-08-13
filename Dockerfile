# ==============================================================================
# InfraNova AI — Production Multi-Stage Dockerfile
# Builds the React frontend and serves it alongside the FastAPI backend.
#
# Usage:
#   docker build -t infranova-ai .
#   docker run -p 8000:8000 -v ./outputs:/app/outputs infranova-ai
#
# For CUDA support, override the build arg:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t infranova-ai .
# ==============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build the React frontend
# ---------------------------------------------------------------------------
FROM node:18-alpine AS frontend-build

WORKDIR /web

COPY web/package.json web/package-lock.json* ./
RUN npm install

COPY web/ .
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Python runtime with FastAPI + built frontend
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch (CPU by default), then remaining dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install torch torchvision torchaudio --index-url ${TORCH_INDEX_URL} && \
    python -m pip install -r /app/requirements.txt

# Copy entire project source
COPY . /app

# Copy built React assets into a static directory served by FastAPI
COPY --from=frontend-build /web/dist /app/web/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
