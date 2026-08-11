# InfraNova AI — Rollback Guide

If a model deployment, config change, or codebase update causes critical failures in production, follow these procedures to restore a stable state.

## 1. Model Checkpoint Rollback

If a newly trained model performs worse than expected in production (e.g., severe color artifacting or hallucinations).

1. **Locate Previous Best**: All historical models should be stored locally or on Kaggle. 
2. **Restore**: Copy the stable `.pth` file over `outputs/models/best/pix2pix_landsat_best.pth`.
3. **Restart API**: The FastAPI backend caches the model in memory upon the first request. You *must* restart the Uvicorn worker to load the new weights.
   ```bash
   # Find the uvicorn process and kill it, then restart:
   pkill -f uvicorn
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

## 2. Web UI Rollback

If the React frontend breaks (e.g., slider stops working, CSS layouts break on mobile).

1. **Revert Git**: Find the last stable commit touching the `web/` directory.
   ```bash
   git checkout <stable_commit_hash> -- web/
   ```
2. **Rebuild**:
   ```bash
   cd web
   npm install
   npm run build
   ```

## 3. Training Config Rollback

If a new loss function weight or architecture change causes training to crash or gradients to explode (e.g., NaN loss).

1. **Restore Config**: Overwrite `configs/config.yaml` with the known good state.
2. **Delete Corrupted Checkpoints**: If training crashed, the `latest/pix2pix_landsat_latest.pth` might be corrupted with NaN weights. Delete it so the trainer doesn't auto-resume from a poisoned state.
   ```bash
   rm outputs/models/latest/pix2pix_landsat_latest.pth
   ```
3. **Restart Training**: Run `train_landsat.py` again. It will fall back to `best.pth` or start from scratch if neither exists.

## 4. Dataset Quarantine Rollback

If `validate_patch_dataset.py` was overly aggressive and quarantined good data:

1. **Move Files Back**:
   ```bash
   mv data/landsat9/splits/quarantine/* data/landsat9/splits/train/
   ```
2. **Dry-Run Validation**: Adjust the variance thresholds in `validate_patch_dataset.py` and run it again to ensure it only targets actual bad patches.
