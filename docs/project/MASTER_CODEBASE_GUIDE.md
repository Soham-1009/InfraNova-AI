# INFRA NOVA AI — MASTER CODEBASE GUIDE

Welcome to the definitive documentation system for InfraNova AI. This system was generated via a comprehensive codebase audit, ensuring that it reflects the actual implementation rather than outdated assumptions.

Whether you are an AI session picking up where a human left off, a new developer onboarding, or a researcher looking to improve the model, use this guide as your starting point.

---

## 🧭 Directory

### System & Architecture
- [**Project Overview**](PROJECT_OVERVIEW.md) — The 60-second, 10-minute, and technical deep-dives into what this project is.
- [**Architecture**](ARCHITECTURE.md) — Mermaid diagrams showing how the ML, data, and web components interact.
- [**Execution Flow**](FLOW.md) — Step-by-step traces of the data pipeline, training loop, and inference.
- [**Codebase Map**](CODEBASE_MAP.md) — A file-by-file breakdown of responsibilities and modification risks.
- [**System Constraints**](CONSTRAINTS.md) — Hard hardware, mathematical, and data boundaries you must respect.

### Machine Learning
- [**ML Concepts**](ML_CONCEPTS.md) — Explanations of Pix2Pix, PatchGAN, Dynamic U-Nets, and various losses.
- [**Model Architecture**](MODEL_ARCHITECTURE.md) — Layer-by-layer details of the Generator and Discriminator.
- [**Losses**](LOSSES.md) — How the 5 individual loss components are calculated and weighted.
- [**Training**](TRAINING.md) — Details on AMP, multi-GPU DataParallel, and the optimization loop.
- [**Inference**](INFERENCE.md) — Preprocessing, Test-Time Augmentation (TTA), and postprocessing.
- [**Data Pipeline**](DATA_PIPELINE.md) — How Landsat 9 data is downloaded, processed, audited, and augmented.
- [**Experiments**](EXPERIMENTS.md) — History of training runs and current best metrics.

### Engineering & Deployment
- [**API and Web**](API_AND_WEB.md) — Architecture of the FastAPI backend and React frontend.
- [**Deployment**](DEPLOYMENT.md) — Docker configurations and model export strategies (ONNX/TorchScript).
- [**Configuration**](CONFIGURATION.md) — How the YAML config system and CLI overrides work.
- [**Bugs**](BUGS.md) — Known issues, stale configurations, and missing files.
- [**Decisions**](DECISIONS.md) — The architectural "why" (e.g., Why Pix2Pix? Why not Streamlit?).
- [**Implementation Status**](IMPLEMENTATION_STATUS.md) — The single source of truth mapping features to code and tests.

### Operations & Collaboration
- [**AI Handover**](AI_HANDOVER.md) — The state of the project tailored for an AI assistant to resume work instantly.
- [**Developer Guide**](DEVELOPER_GUIDE.md) — Environment setup, code quality standards, and API/UI modification guides.
- [**Collaborator Guide**](COLLABORATOR_GUIDE.md) — High-level onboarding for data scientists and ML researchers.
- [**Test Plan**](TEST_PLAN.md) — Strategy for unit testing, smoke testing, and UI verification.
- [**Rollback**](ROLLBACK.md) — Procedures for recovering from bad models, crashed training, or broken UI.
- [**Documentation Audit**](DOCUMENTATION_AUDIT.md) — The policy for keeping these documents accurate.

---

## 🚨 Critical Warnings for Newcomers

1. **Docker is Stale**: The `Dockerfile` and `docker-compose.yml` still try to run the deleted Streamlit app. Do not use them until they are rewritten for FastAPI/React.
2. **`image_size` is Strict**: The PyTorch Dataset will crash if `.npy` patches do not exactly match the `image_size` in the config. Do not change `image_size` to 256 without completely regenerating the dataset from Earth Engine.
3. **Losses Directory is Fake**: The `src/losses/` directory is an empty shell. All actual loss functions live in `src/training/losses.py`.

---
> **End of Master Guide**. To begin work, open the [AI Handover](AI_HANDOVER.md) or [Developer Guide](DEVELOPER_GUIDE.md).
