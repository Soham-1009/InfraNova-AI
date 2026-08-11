# InfraNova AI — Collaborator Guide

Welcome to the InfraNova AI project! This guide is for data scientists, ML engineers, and researchers looking to understand the project at a high level and contribute to the model.

## 1. Project Philosophy

- **The Source of Truth is the Code**: Documentation can drift. Always verify model architectures by reading `src/models/` and configurations by checking `configs/config.yaml`.
- **No Magic Numbers**: If a parameter affects the model or dataset, it belongs in the YAML config, not hardcoded deep in a `.py` file.
- **Reproducibility**: We use seeds (`42`) and avoid stochastic operations where possible outside of explicit augmentation. Checkpoints save their config hash to ensure we know exactly what parameters produced them.

## 2. Quick Start

1. Clone the repository.
2. Create a virtual environment (`python -m venv venv`) and activate it.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the API: `uvicorn api.main:app`.
5. Run the UI (requires Node.js): `cd web && npm install && npm run dev`.

## 3. How to Contribute to the Model

If your goal is to improve the SSIM score (currently ~0.43):

1. **Understand the Baseline**: Read `docs/project/EXPERIMENTS.md` and `docs/project/LOSSES.md`. Know what has already been tried.
2. **Modify the Config**: Try tweaking `lambda_chroma` or switching `normalization.mode` to `global` in `configs/config.yaml`.
3. **Run a Local Smoke Test**: `python src/training/train_landsat.py --config configs/config.yaml --overrides epochs=1 batch_size=2`. Ensure your changes didn't break the forward/backward passes.
4. **Train on GPU**: Push your changes, pull them into your Kaggle environment, and run `notebooks/MAIN.ipynb`.
5. **Commit the Checkpoint**: If SSIM improves, download `pix2pix_landsat_best.pth` and replace the one in `outputs/models/best/`. Update the `EXPERIMENTS.md` log.

## 4. Git Workflow

- **Main Branch**: Only for stable code. The web UI and API must be functional.
- **Feature Branches**: Branch off `main` for experiments (e.g., `feat/add-cyclegan-loss` or `fix/docker-build`).
- **Pull Requests**: Require a quick summary of what changed and proof that the unit tests pass (`pytest tests/`).

## 5. Getting Help

If you're stuck, refer to the `docs/project/MASTER_CODEBASE_GUIDE.md` to find the exact document you need, or check the `BUGS.md` to see if you hit a known issue.
