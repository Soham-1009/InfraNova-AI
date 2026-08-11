# InfraNova AI — Test Plan

This document outlines the testing strategy to ensure the ML pipeline and Web API function correctly after any code changes.

## 1. Automated Testing (Unit Tests)

The repository uses `pytest` for automated unit testing. All tests are located in the `tests/` directory.

### 1.1 Running Tests
```bash
pytest tests/ -v
```

### 1.2 Core Test Suites
- **`test_core_behaviour.py`**: Verifies that the generator and discriminator can perform forward and backward passes without shape mismatch errors or NaN gradients.
- **`test_generator_regression.py`**: Compares the new Dynamic U-Net output shapes against the legacy 8-block U-Net to ensure architectural parity.
- **`test_losses.py`**: Validates the math of custom losses (e.g., ensuring SSIM returns 1.0 for identical images, and L1 returns 0.0).
- **`test_dataset.py`**: Ensures the dataloader outputs exactly `[B, 1, 128, 128]` for IR and `[B, 3, 128, 128]` for RGB, and verifies that values are strictly in the `[-1.0, 1.0]` range.
- **`test_checkpoint.py`**: Verifies that saving a model and loading it back yields identical weights.

## 2. Integration Testing (Smoke Tests)

Before kicking off a 6-hour Kaggle training run, you must verify the integration of the dataset, trainer, and model.

### 2.1 The Training Smoke Test
Run the full training loop for exactly 1 epoch with a tiny batch size.
```bash
python src/training/train_landsat.py --config configs/config.yaml --overrides epochs=1 batch_size=4
```
**Expected Outcome**: The script completes without OOM errors, prints "Epoch 1/1", runs the validation loop, and saves a `latest.pth` checkpoint to `outputs/models/latest/`.

## 3. End-to-End System Testing

After modifying the React UI or FastAPI backend, test the end-to-end user experience.

### 3.1 API Health
```bash
curl http://localhost:8000/health
```
**Expected Outcome**: `{"status": "ok", "model_loaded": false}`

### 3.2 Visual UI Verification
1. Start both servers (`uvicorn api.main:app` and `npm run dev`).
2. Open `http://localhost:5173` in a browser.
3. Drag and drop a sample thermal image from `demo/assets/samples/`.
4. **Verification Checklist**:
   - [ ] Loading spinner appears.
   - [ ] Both `/colorize` and `/thermal-preview` network requests succeed (200 OK).
   - [ ] The "before/after" slider functions smoothly.
   - [ ] The UI does not spawn vertical scrollbars (fits within 100vh).
   - [ ] The footer text "Powered by PyTorch & Pix2Pix GAN" is fully visible.

## 4. Continuous Integration (Future)

Currently, the `.github/workflows/` directory is empty. 
A future CI pipeline should:
1. Run `pytest tests/` on every Pull Request.
2. Run `flake8` or `ruff` to enforce PEP8 styling.
3. Attempt to build the Docker container (once the Dockerfile is fixed).
