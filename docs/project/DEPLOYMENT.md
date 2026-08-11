# InfraNova AI — Deployment Guide

This document covers deployment configurations, known issues, and export strategies for the InfraNova AI model.

## 1. Docker Deployment (CURRENTLY BROKEN)

The repository contains a `Dockerfile` and `docker-compose.yml`. **These files are currently stale and broken.**

### 1.1 The Issue
The Dockerfiles were originally written for a Streamlit app located at `demo/streamlit_app.py`. During the migration to the React/FastAPI stack, `streamlit_app.py` was deleted, but the Dockerfiles were not updated. If built today, the container will crash on startup because it attempts to run a non-existent file.

### 1.2 Required Fixes
To deploy the current stack via Docker, the configuration must be rewritten to support a multi-stage or multi-container architecture:
1. **Container 1 (Backend)**: Python 3.11 image running FastAPI via Uvicorn on port 8000.
2. **Container 2 (Frontend)**: Node image building the React app, served via an Nginx alpine image on port 80 or 443.

Alternatively, the React build (`web/dist/`) could be served directly by FastAPI via `StaticFiles` in a single-container deployment.

## 2. Model Export

Currently, the production API loads raw PyTorch checkpoints (`.pth`). For higher performance deployment (e.g., C++ servers, Edge devices, or TensorRT), the codebase provides export scripts.

### 2.1 ONNX Export
- **Script**: `scripts/deployment/export/export_onnx.py`
- **Output**: `pix2pix_generator.onnx`
- **Benefits**: Framework-agnostic. Can be run via ONNXRuntime on CPU with significantly lower overhead than PyTorch.
- **Status**: Script is implemented but the resulting ONNX file has not been integrated into the API.

### 2.2 TorchScript Export
- **Script**: `scripts/deployment/export/export_model.py`
- **Output**: `pix2pix_generator.pt` (TorchScript)
- **Benefits**: Can be loaded in C++ via LibTorch without a Python dependency.
- **Status**: Implemented but untested.

## 3. Deployment Constraints

1. **Memory Requirements**: The generator is relatively lightweight (~15MB weights). CPU inference requires < 500MB RAM.
2. **Hardware Constraints**: GPU is NOT required for inference. A modern CPU can process a 128x128 patch in ~100-200ms.
3. **Weight Storage**: Model weights are NOT tracked in git due to size limits. They must be downloaded manually or mounted via volumes (as configured in the current `docker-compose.yml`).
