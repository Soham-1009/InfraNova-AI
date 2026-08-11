# InfraNova AI — AI Handover Document

> **Last updated**: 2026-08-11 | **Author**: Automated codebase analysis

---

## CURRENT STATE

| Attribute | Value |
|-----------|-------|
| **Project** | InfraNova AI |
| **Purpose** | Convert Landsat 9 thermal infrared (TIR) satellite imagery into photorealistic RGB-like satellite imagery |
| **Current Objective** | Model improvement (SSIM currently ~0.43) |
| **Architecture** | Pix2Pix conditional GAN (Dynamic U-Net generator + PatchGAN discriminator) |
| **Model** | `Pix2Pix` with `GeneratorUNetDynamic` (depth=7 for 128×128) + `PatchDiscriminator` with spectral normalization |
| **Dataset** | Landsat 9 paired patches: `tir_100m.npy` (128×128) → `rgb_100m.npy` (3×128×128). ~13,384 train, ~1,630 val samples |
| **Training Status** | Trained to epoch 226/250 on Kaggle (2× Tesla T4). Training ran across 6 experiment sessions |
| **Best Metrics** | SSIM: 0.4290, PSNR: 13.27, val_sat_ratio: 0.438, val_lab_error: 24.03 |
| **Best Checkpoint** | `outputs/models/best/pix2pix_landsat_best.pth` |
| **Deployment Status** | React + FastAPI web app (localhost). Docker config exists but references obsolete Streamlit |

---

## IMPORTANT FILES

| File | Role |
|------|------|
| `configs/config.yaml` | Master training configuration |
| `src/models/pix2pix/pix2pix.py` | Pix2Pix wrapper (generator + discriminator) |
| `src/models/pix2pix/generator_dynamic.py` | Dynamic U-Net generator (production) |
| `src/models/pix2pix/generator.py` | Legacy 8-block U-Net (256×256 only, not used) |
| `src/models/pix2pix/discriminator.py` | PatchGAN + MultiScale discriminator |
| `src/training/losses.py` | All loss functions (GAN, L1, Perceptual, SSIM, Chroma, Feature Matching) |
| `src/training/trainer.py` | Training loop |
| `src/training/train_landsat.py` | Training entry point |
| `src/datasets/landsat9_dataset.py` | PyTorch Dataset for Landsat 9 patches |
| `src/inference/landsat_inference.py` | Production inference engine |
| `demo/inference.py` | Web-app inference engine (used by API) |
| `demo/utils.py` | Preprocessing/postprocessing for inference |
| `api/main.py` | FastAPI backend |
| `web/src/App.jsx` | React frontend |
| `web/src/index.css` | UI design system |

---

## CURRENT CONFIG (from `configs/config.yaml`)

| Parameter | Value |
|-----------|-------|
| image_size | 128 |
| input_channels | 1 |
| output_channels | 3 |
| generator | dynamic |
| multi_scale_disc | false |
| epochs | 250 |
| batch_size | 128 |
| lr | 0.0002 |
| decay_start_epoch | 230 |
| patience | 40 |
| amp | true |
| grad_clip | 1.0 |
| lambda_adv | 1.0 |
| lambda_l1 | 10.0 |
| lambda_perc | 10.0 |
| lambda_ssim | 5.0 |
| lambda_chroma | 2.0 |
| lambda_feat | 0.0 |
| gan_mode | bce |
| normalization | local (per-sample percentile stretch) |

---

## RECENT CHANGES

1. **Migrated web stack**: Removed `streamlit_app.py`, built React + FastAPI architecture
2. **Fixed model loading**: Stripped `.module.` prefix from DataParallel checkpoint keys; pass `image_size=128` to `Pix2Pix`
3. **Fixed `batch_inference.py`**: Renamed `tta` parameter to `use_tta`
4. **Fixed footer layout**: Reduced padding so footer fits viewport
5. **Fixed `.gitignore`**: Removed blanket `*.json` rule; added React/Node artifacts
6. **Updated GitHub links**: Corrected to `Soham-1009/InfraNova-AI`
7. **Updated README**: Replaced Streamlit docs with React+FastAPI Web App section

---

## KNOWN BUGS

1. **Docker config stale**: `Dockerfile` and `docker-compose.yml` still reference `streamlit` and port 8501. They need updating for FastAPI backend
2. **`src/losses/` directory is empty**: All loss implementations are actually in `src/training/losses.py`. The empty directory is misleading
3. **`LandsatColorizationInference` in `src/inference/landsat_inference.py`** validates `image_size` as multiple of 256 (`line 58`), but the model is trained at 128. The production API uses `demo/inference.py` instead, which validates as multiple of 128
4. **`config_smoke.yaml` mentioned in README does not exist**: Only `config.yaml` is present in `configs/`

---

## KNOWN LIMITATIONS

1. **SSIM ~0.43**: Model produces plausible but not photorealistic outputs
2. **Single-channel input**: Only uses TIR band ST_B10. Adding more spectral bands could improve results
3. **128×128 resolution only**: Both the training patches and the model are configured for 128×128. Upscaling to 256×256 requires retraining
4. **Local normalization**: Per-sample percentile stretching means normalization varies per-image. Global normalization infrastructure exists but is unused
5. **No YOLO/detection**: `src/detection/` is an empty module with only `__init__.py`
6. **No CI/CD**: `.github/workflows/` directory is empty

---

## DO NOT CHANGE

| Constraint | Reason |
|------------|--------|
| `image_size: 128` in config | All training patches are 128×128. Changing this without rebuilding patches will crash the dataset loader (strict shape validation) |
| Checkpoint format (`model_state_dict` key) | Changing the key name breaks `load_checkpoint()` |
| `.module.` stripping in inference | Required because checkpoints were saved under `DataParallel` (multi-GPU training on Kaggle) |
| `[-1, 1]` normalization range | Generator output uses `Tanh`. All preprocessing and postprocessing assumes this range |
| Region-based dataset splits | Patches from the same region must stay in the same split (train/val/test) to prevent data leakage |

---

## CURRENT TASK

Model improvement — push SSIM above 0.43.

## NEXT STEPS

1. Download diverse Landsat 9 regions (desert, forest, ocean, snow) to improve generalization
2. Fine-tune at 256×256 resolution using progressive training
3. Update Docker config for FastAPI backend
4. Add CI/CD pipeline

---

## STATUS SUMMARY

| Category | Status |
|----------|--------|
| Data pipeline | ✅ DONE |
| Model training | ✅ DONE (epoch 226, SSIM 0.43) |
| Production inference | ✅ DONE |
| React web app | ✅ DONE |
| FastAPI backend | ✅ DONE |
| Git/GitHub | ✅ DONE |
| Docker | ⚠️ STALE (references Streamlit) |
| CI/CD | ❌ NOT STARTED |
| Model improvement | 🔄 IN PROGRESS |

---

## VALIDATION REQUIRED

- [ ] Run `pytest tests/` to verify test suite passes
- [ ] Verify Docker build after updating Dockerfile
- [ ] Run full training to completion (250 epochs) and compare SSIM
