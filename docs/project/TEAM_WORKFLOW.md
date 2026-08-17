# InfraNova AI Team Workflow

This document defines the roles and responsibilities for the 4-person team working on InfraNova AI. The architecture has been intentionally decoupled to ensure that all 4 roles can develop in parallel with zero merge conflicts.

---

## 1. Data Engineer (Data Pipeline & Preprocessing)
**Role**: Handles raw Landsat 9 data ingestion and patching.

- **Primary Ownership**:
  - `src/datasets/`
  - `notebooks/`
  - Kaggle Datasets (`infranova-ai-full`)
- **Key Responsibilities**:
  - Download and filter GeoTIFFs from Google Earth Engine.
  - Maintain the patching scripts in `notebooks/`.
  - Push the updated `splits.zip` (train/val/test data) to Kaggle.
  - Validate image ranges `[-1, 1]` and filter out anomalies (clouds, water bodies).

## 2. Core ML Engineer (Model Architecture & Training)
**Role**: Builds and trains the Pix2Pix Generative Adversarial Network.

- **Primary Ownership**:
  - `src/models/` (Generator, Discriminator)
  - `src/training/` (Trainer, Losses, Schedulers)
  - `configs/config.yaml`
- **Key Responsibilities**:
  - Adjust hyper-parameters (learning rates, batch sizes, epochs).
  - Modify the Dynamic U-Net Generator and PatchGAN Discriminator.
  - Implement and balance the composite objective functions (L1, cGAN, Perceptual, SSIM).
  - Train the model using Kaggle T4x2 environments (`notebooks/KAGGLE.ipynb`).

## 3. Backend & MLOps Engineer (API & Deployment)
**Role**: Wraps the trained AI into a fast, scalable backend service.

- **Primary Ownership**:
  - `api/` (FastAPI)
  - `src/inference/` (Optimized inference pipeline)
  - `Dockerfile` & `docker-compose.yml`
  - `.github/workflows/`
- **Key Responsibilities**:
  - Build the FastAPI server endpoints.
  - Ensure the `InferenceEngine` efficiently processes images.
  - Export PyTorch weights to optimized formats (ONNX/TorchScript).
  - Manage containerization and CI/CD actions.

## 4. Frontend Developer (Web Application)
**Role**: Builds the interface users interact with.

- **Primary Ownership**:
  - `web/` (React SPA)
- **Key Responsibilities**:
  - Design and style the user interface.
  - Connect the React application to the FastAPI backend.
  - Implement dynamic UI features (e.g., interactive before/after image sliders).
  - Handle loading states, error boundaries, and responsiveness.
