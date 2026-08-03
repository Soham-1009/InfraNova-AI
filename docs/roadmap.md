# 🛰️ InfraNova AI: Landsat 9 TIR Colorization Roadmap

**Objective:** End-to-End Pipeline for Landsat 9 TIR → RGB satellite image colorization

---

### ✅ Phase 1: Data Pipeline

* Configured PyTorch DataLoaders to accept 1-channel Landsat 9 Band 10 (TIRS-2) data.
* Built 128×128 paired patches with telemetry-driven filtering (NoData/Blank rejection).
* Implemented region-level train/val/test splits to prevent geographic leakage.

### ✅ Phase 2: Training

* Launched the Pix2Pix training loop on the 100m TIR → 100m RGB task.
* Implemented deterministic checkpointing with full optimizer state for clean resume.
* Validated 10-epoch stability test: Generator/Discriminator losses stable, SSIM/PSNR trending upward.

### ✅ Phase 3: Evaluation and Tuning

* Extracted LPIPS, PSNR, and SSIM metrics.
* Created inference consistency tests (`scripts/evaluation/test_inference.py`).
* Validated checkpoint reload consistency (0.0 float diff).

### ✅ Phase 4: Streamlit Demo

* Built full-featured Streamlit demo with dark mode, TTA toggle, contrast enhancement, and download.
* Implemented `@st.cache_resource` for model loading to ensure responsive UI.
* Added pre-loaded sample cards for instant testing.

### ✅ Phase 5: Deployment Preparation

* Created Kaggle smoke-test configuration (`configs/config_smoke.yaml`).
* Prepared `kaggle_smoke_test.md` deployment guide.
* Generated model card via `scripts/deployment/generate_model_card.py`.
* Docker setup for local demo deployment.

---

### 🖥️ Demo Strategy

The Streamlit UI tells a technical story of capability, not just image processing.

* **The Input Zone:** Users upload a Landsat 9 TIR TIFF or select from pre-loaded scenarios (e.g., Urban Heat Island, Coastal Zone, Agricultural Basin).
* **The Pipeline View:**
  * *Column 1:* Raw TIR input.
  * *Column 2:* Synthesized RGB output.
* **The Object Detection Toggle:** A switch labeled "Run Object Detection (YOLO)". When enabled, it overlays bounding boxes and confidence scores on both the TIR and RGB images, proving the colorization adds machine-readable semantic value.
* **Export Options:** Download the synthesized RGB for further analysis.

---

### 📊 Presentation Structure

**Slide 1: Title**
* *Content:* "InfraNova AI: TIR Colorization for Earth Observation"
* *Visual:* Clean, high-contrast side-by-side of a TIR patch and our RGB output.

**Slide 2: The Problem**
* *Content:* Night-time and thermal imaging is monochrome and lacks semantic texture, limiting automated disaster and environmental monitoring.

**Slide 3: Solution Architecture**
* *Content:* Dynamic U-Net generator with PatchGAN discriminator. Combined L1 + adversarial + perceptual + SSIM + chroma losses.
* *Visual:* Block diagram of the neural network topology.

**Slide 4: Innovation & Differentiation**
* *Content:* Object-detection validation loop. We don't just optimize for human eyes; we optimize for downstream machine learning interpretability.

**Slide 5: Results & Metrics**
* *Content:* Display SSIM, PSNR, LPIPS, and YOLO mAP improvements.
* *Visual:* 3×3 grid of the hardest test patches (e.g., distinguishing a dark river from a dark road).

**Slide 6: Technical Stack**
* *Content:* Python, PyTorch, Dynamic U-Net, Kaggle GPU training, Streamlit demo, Docker deployment.

**Slide 7: Demo Screenshots**
* *Content:* Clean, high-res captures of the Streamlit interface demonstrating the user flow.

**Slide 8: Future Work**
* *Content:* Multi-spectral fusion, higher resolution (256×256), temporal context conditioning, production API deployment.

---

### 🎬 Demo Video Script (90 Seconds)

*(Visual: Screen recording starts on the Streamlit landing page.)*

**[0:00–0:15] The Hook:**
"Welcome to InfraNova AI. Satellite platforms capture critical thermal infrared data at night, but these monochrome images are difficult for both humans and AI to interpret. Today, we are solving that."

**[0:15–0:40] The Transformation:**
*(Visual: Click a preloaded sample card. The UI processes the image.)*
"Here we load a raw Landsat 9 Band 10 thermal image. Watch as our Dynamic U-Net Pix2Pix network synthesizes photorealistic RGB textures from the thermal input."

**[0:40–1:10] The Proof (Object Detection):**
*(Visual: Toggle the Object Detection button. Bounding boxes appear.)*
"But the true value is machine interpretability. When we run a standard YOLO detector on the raw TIR, confidence is low. When applied to our synthesized RGB output, vehicle and structural detection confidence surges. We've converted unreadable thermal noise into actionable data."

**[1:10–1:30] The Close:**
"InfraNova AI provides a deployable, end-to-end framework for 24/7 earth observation. Thank you."

---

### 🚀 Project Narrative

Use cases this project directly addresses:

* **Urban Heat Island & Micro-climate Tracking:** Since TIR explicitly measures heat, colorizing these thermal maps helps urban planners track temperature anomalies in expanding metropolitan areas.
* **Disaster Response:** During a flood or landslide under heavy cloud cover or at night, TIR is the only reliable sensor. Colorizing it instantly helps emergency responders identify roads vs. water bodies.
* **Agricultural Monitoring:** Thermal imagery reveals crop stress, irrigation patterns, and soil moisture — all more interpretable when presented as natural-looking RGB.

---

### 🧠 Technical Differentiation

What separates InfraNova AI from other approaches:

1. **Real Satellite Data:** Uses actual multi-band Landsat 9 Level-2 data, not ground-level synthetic datasets like FLIR.
2. **The "Delta" Approach:** Success is measured not just through SSIM metrics, but by the improvement in a separate AI model (YOLO) acting on our outputs.
3. **Production-Ready Code:** Repository is structured with clear separation of concerns (`configs/`, `src/`, `scripts/`, `demo/`, `tests/`).
4. **Dynamic Architecture:** U-Net depth adapts automatically to input dimensions, supporting multiple resolutions without code changes.

---

### 🛡️ Risk Mitigation

* **Training Fails (Mode Collapse):** Fall back to earlier checkpoint weights. An imperfect colorizer with a working demo is better than a broken model with no demo.
* **Demo Breaks During Testing:** Have pre-rendered outputs cached and a pre-recorded video as backup.

---

### ✅ Final Checklist

* [ ] **GitHub Repository Public:** Check in an Incognito window.
* [ ] **`README.md` Polished:** Must include architecture diagram and setup instructions.
* [ ] **Model Weights Uploaded:** Do not push `.pth` files to GitHub if they exceed 100MB; use Hugging Face or a Kaggle Dataset.
* [ ] **Code Cleansed:** Remove any hardcoded local paths (e.g., `C:/Users/...`) and replace with relative paths.

---

### 🔭 Future Strategy

* **Open Source Release:** Package the inference module as a standalone PyPI library for the remote sensing community.
* **Academic Publication:** If final SSIM breaks 0.70 on Landsat 9 data, draft a paper focusing on downstream object detection accuracy improvements.
* **Higher Resolution:** Scale to 256×256 and explore multi-spectral input conditioning.
* **Temporal Context:** Add time-of-day and season metadata as auxiliary conditioning to improve disambiguation.

---

**Status:** All phases completed. Pipeline is frozen and validated for Kaggle deployment via `configs/config_smoke.yaml`.
