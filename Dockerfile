# ==============================================================================
# InfraNova AI — Production Multi-Stage Dockerfile
# Builds the React frontend and serves it alongside the FastAPI backend.
#
# Usage:
#   docker build -t infranova-ai .
#   docker run -p 8000:8000 -v ./outputs:/app/outputs:ro infranova-ai
#
# For CUDA support, override the build arg:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t infranova-ai .
# ==============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build the React frontend
# ---------------------------------------------------------------------------
FROM node:20-alpine AS frontend-build

ARG VITE_API_URL=""
ENV VITE_API_URL=${VITE_API_URL}

WORKDIR /web

COPY web/package.json web/package-lock.json* ./
RUN npm ci

COPY web/ .
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Python runtime with FastAPI + built frontend
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install minimal system dependencies for OpenCV and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy production requirements first for optimal layer caching
COPY requirements-prod.txt /app/requirements-prod.txt

# Install PyTorch (CPU by default), then production runtime dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install torch torchvision --index-url ${TORCH_INDEX_URL} && \
    python -m pip install -r /app/requirements-prod.txt

# Copy application source (excluding files in .dockerignore)
COPY . /app

# Copy built React assets from Stage 1 into static directory served by FastAPI
COPY --from=frontend-build /web/dist /app/web/dist

# Ensure outputs directory structure exists and configure non-root user
RUN mkdir -p /app/outputs/best && \
    useradd -u 1000 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
