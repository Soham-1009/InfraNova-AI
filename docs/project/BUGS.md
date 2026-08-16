# Known Bugs

*Currently, there are no known functional or structural bugs in the codebase.*

## Recently Resolved

- **BUG-001 (Matplotlib memory leak):** The figure variable was assigned to `_fig` in `trainer.py` so the `finally` block's `plt.close(fig)` never ran, leaking figures. Fixed by renaming `_fig` to `fig`.
- **BUG-002 (Docker config mismatch):** The API service in `docker-compose.yml` was configured to use `api/Dockerfile` instead of the root `Dockerfile`, causing `ModuleNotFoundError`s without bind mounts. Fixed.
- **BUG-003 (Route shadowing):** Static files were mounted at `/` via FastAPI lifespan which could shadow API routes in production. Fixed by moving `app.mount()` to the end of `api/main.py` so it catches routes only after API endpoints.
- **BUG-004 (Integration test pollution):** `tests/test_training_integration.py` was writing test checkpoints to the project tree instead of a temporary directory. Fixed by using `tempfile` and `tmp_path`.
- **BUG-005 (AMP scaler logic):** In `trainer.py`, `GradScaler.update()` was shared for both generator and discriminator steps. Fixed per PyTorch docs (calling `update` after each `step`).
- **BUG-006 (Config inconsistency):** The default `image_size` in `LandsatColorizationInference` was 256 while `demo/inference.py` was 128. Fixed to 128 to match the trained weights.
- **BUG-007 (Batch inference performance):** `predict_batch` in `demo/inference.py` processed images sequentially in a loop. Fixed to perform true vectorized batched inference (and batched TTA).
- **Frontend CSS Grid:** The slider image in `web/src/index.css` was squished. Fixed using modern CSS Container Queries (`100cqw`, `100cqh`).
- **Linter Errors (Ruff):** Missing `from exc` in the `except` blocks of `api/main.py`. This is fully fixed.
- **CI Test Failure:** The `test_training_integration.py` was failing in the GitHub Actions Linux runner because the `experiments/` directory did not exist. Fixed by explicitly creating the directory inside the test before attempting to save the checkpoint.

---
_Note: All automated tests (`pytest`), `ruff` static analysis, and the `Vite` production frontend builds are currently passing successfully!_
