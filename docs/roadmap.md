Here is the comprehensive strategic playbook to transition InfraNova AI to the Landsat 9 TIR specification and finalize the submission for the ISRO BAH 2026 hackathon.

---

# 🛰️ InfraNova AI: Landsat 9 TIR Colorization Roadmap

**File Location:** `docs/landsat9_roadmap.md`
**Target:** ISRO BAH 2026 Submission
**Objective:** End-to-End Pipeline for 200m TIR $\rightarrow$ 100m TIR $\rightarrow$ 100m RGB

---

### 📅 1. EXECUTION TIMELINE (8-DAY SPRINT)

**Day 1: Training Setup & Data Alignment**

* **Deliverables:** Reconfigure PyTorch DataLoaders to accept 1-channel Landsat 9 Band 10 (TIRS-2) data instead of Sentinel-2. Verify all 1800 patches are strictly paired.
* **Time Estimate:** 4 Hours.
* **Risk:** Landsat 9 TIRS-2 has absolute radiometric temperature ranges differing from Sentinel-2 NIR.
* **Backup:** Hard-cap pixel intensity normalization between 270K and 320K (typical Earth surface temperatures) to prevent vanishing gradients.

**Day 2: Training Execution (Colorization)**

* **Deliverables:** Launch the Pix2Pix training loop on the 100m TIR $\rightarrow$ 100m RGB task.
* **Time Estimate:** 8–10 Hours (Background execution).
* **Risk:** Colab/Kaggle timeout during training.
* **Backup:** Save optimizer state and model weights every 10 epochs to Google Drive.

**Day 3: Evaluation and Tuning**

* **Deliverables:** Extract LPIPS, PSNR, and SSIM metrics. Run initial qualitative tests.
* **Time Estimate:** 3 Hours.
* **Risk:** The model produces "muddy" or desaturated outputs.
* **Backup:** Temporarily halve the L1 structural loss weight to force the GAN to take more risks with color generation.

**Day 4: Super-Resolution (SR) Module Integration**

* **Deliverables:** Implement the 200m $\rightarrow$ 100m TIR upscaling. Use a lightweight pre-trained Real-ESRGAN or standard bicubic interpolation if VRAM is tight, feeding the output directly into the colorizer.
* **Time Estimate:** 5 Hours.
* **Risk:** The SR artifacts confuse the Pix2Pix colorizer.
* **Backup:** Apply a slight Gaussian blur to the SR output before passing it to the colorization module to smooth out jagged upsampling edges.

**Day 5: Update Streamlit Demo UI**

* **Deliverables:** Refactor the UI to process the three-step pipeline (Upload $\rightarrow$ SR $\rightarrow$ Colorize). Integrate YOLOv10 for the final downstream task.
* **Time Estimate:** 6 Hours.
* **Risk:** The UI becomes unresponsive due to the heavy triple-inference pipeline.
* **Backup:** Use `@st.cache_data` aggressively and restrict test inputs to 256x256 crops during live UI interactions.

**Day 6: Pitch Deck & Demo Video**

* **Deliverables:** Finalize the 8-slide presentation and record the 90-second video.
* **Time Estimate:** 5 Hours.

**Day 7: Polish and Test**

* **Deliverables:** Conduct a full mock presentation. Test the Streamlit app with edge-case images (e.g., heavily clouded TIR images).
* **Time Estimate:** 3 Hours.

**Day 8: Final Submission**

* **Deliverables:** Upload weights to GitHub/Hugging Face. Submit the technical PDF and video to the Hack2Skill platform.
* **Time Estimate:** 2 Hours (Do this 6 hours before the actual deadline).

---

### 🖥️ 2. DEMO STRATEGY UPDATE

The Streamlit UI must tell a technical story of mission capability, not just image processing.

* **The Input Zone:** Allow users to upload a 200m Landsat 9 TIR TIFF or select from 3 pre-loaded scenarios (e.g., Urban Heat Island, Coastal Zone, Agricultural Basin).
* **The Pipeline View:** * *Column 1:* Raw 200m TIR.
* *Column 2:* Super-Resolved 100m TIR.
* *Column 3:* Synthesized 100m RGB.


* **The "Mission Value" Toggle:** A switch labeled "Run Object Detection (YOLO)". When flipped, it overlays bounding boxes and confidence scores on both the TIR and RGB images, proving the colorization adds machine-readable semantic value.
* **Export Options:** Provide a button to download the synthesized RGB as a properly banded geoTIFF for GIS software compatibility.

---

### 📊 3. PITCH DECK STRUCTURE

**Slide 1: Title**

* *Content:* "InfraNova AI: TIR Super-Resolution & Colorization for Mission Operations"
* *Visual:* Clean, high-contrast side-by-side of a TIR patch and our RGB output.

**Slide 2: The ISRO Problem**

* *Content:* Night-time and thermal imaging (Landsat 9/INSAT) is monochrome and lacks semantic texture, bottlenecking automated disaster and border monitoring.

**Slide 3: Solution Architecture**

* *Content:* 2-Stage Pipeline. Stage 1: Spatial enhancement (200m $\rightarrow$ 100m). Stage 2: Generative Colorization (100m TIR $\rightarrow$ 100m RGB).
* *Visual:* A block diagram of the neural network topology.

**Slide 4: Innovation & Differentiation**

* *Content:* Highlighting our unique object-detection validation loop. We don't just optimize for human eyes; we optimize for downstream machine learning interpretability.

**Slide 5: Results & Metrics**

* *Content:* Display SSIM, PSNR, and YOLO mAP improvements.
* *Visual:* 3x3 grid of the hardest test patches (e.g., distinguishing a dark river from a dark road).

**Slide 6: Technical Details & Team Rigor**

* *Content:* Highlight the robustness of the engineering stack. Emphasize the full-stack architecture (leveraging Python for the heavy ML lifting with React/Tailwind design principles applied to the Streamlit UI) and your postgraduate AI/ML research context to establish academic credibility.

**Slide 7: Demo Screenshots**

* *Content:* Clean, high-res captures of the Streamlit interface demonstrating the user flow.

**Slide 8: Future Work & ISRO Integration**

* *Content:* Direct pathway to integrating this module into the Bhuvan Geoportal for real-time agricultural and urban heat mapping.

---

### 🎬 4. DEMO VIDEO SCRIPT (90 Seconds)

*(Visual: Screen recording starts on the Streamlit landing page. Confident, steady tech-demo tone. No background music, keep it strictly professional.)*

**[0:00 - 0:15] The Hook:**
"Welcome to InfraNova AI. ISRO's satellite platforms capture critical thermal infrared data at night, but these monochrome images are notoriously difficult for both humans and AI to interpret. Today, we are solving that."

**[0:15 - 0:40] The Transformation:**
*(Visual: Click the 'Urban Nagpur Heat Map' sample. The UI processes the image.)*
"Here we load a raw 200-meter resolution Landsat 9 Band 10 thermal image. Watch as our dual-stage pipeline first applies structural super-resolution to bring it to 100 meters, and then utilizes a custom Pix2Pix generative network to synthesize highly accurate, photorealistic RGB textures."

**[0:40 - 1:10] The Proof (Object Detection):**
*(Visual: Toggle the Object Detection button. Bounding boxes appear.)*
"But the true value is machine interpretability. When we run a standard YOLO detector on the raw TIR, confidence is low. When applied to our synthesized RGB output, vehicle and structural detection confidence surges. We've converted unreadable thermal noise into actionable data."

**[1:10 - 1:30] The Close:**
"InfraNova AI provides a deployable, end-to-end framework ready for integration with platforms like Bhuvan to revolutionize 24/7 earth observation. Thank you."

---

### 🚀 5. ISRO ALIGNMENT NARRATIVE

To win, the judges must see how this fits into their actual workflow:

* **The Bhuvan Geoportal:** Position the colorizer as a potential microservice API for Bhuvan, allowing users to view night-time passes as if they were taken at noon.
* **Urban Heat Island & Micro-climate Tracking:** Since TIR explicitly measures heat, mention how colorizing these thermal maps helps urban planners track temperature anomalies in expanding metropolitan hubs like Nagpur.
* **Disaster Response:** Highlight that during a flood or landslide under heavy cloud cover or at night, TIR is the only reliable sensor. Colorizing it instantly helps emergency responders identify roads vs. water bodies.

---

### 🧠 6. TECHNICAL DIFFERENTIATION

What separates InfraNova AI from the 50 other hackathon teams:

1. **Real Satellite Data:** We are using actual multi-band Landsat 9 Level-2 data, not ground-level synthetic datasets like FLIR.
2. **The "Delta" Approach:** We prove success not just through SSIM metrics, but by measuring the improvement in an entirely separate AI model (YOLO) acting on our outputs.
3. **Production-Ready Code:** The GitHub repository is structured like an enterprise product (`/data`, `/models`, `/notebooks`, `/app`), not a messy single Jupyter notebook.

---

### 🛡️ 7. RISK MITIGATION

* **Training Fails (Mode Collapse):** *Mitigation:* Immediately fall back to the Day 3 weights. An imperfect colorizer with a working Streamlit demo scores higher than a broken model with no demo.
* **Demo Breaks During Judging:** *Mitigation:* Have a pre-recorded video loaded locally on your machine, and cache the absolute best 5 outputs directly into the GitHub repository as static images.
* **Submission Portal Crash:** *Mitigation:* Hackathon portals often crash in the final 30 minutes. The hard rule is to submit the final URL and PDF exactly 6 hours before the official midnight deadline.

---

### ✅ 8. FINAL SUBMISSION CHECKLIST

* [ ] **GitHub Repository Public:** Check in an Incognito window.
* [ ] **`README.md` Polished:** Must include a high-level architecture diagram and standard `pip install -r requirements.txt` instructions.
* [ ] **Model Weights Uploaded:** Do not push `.pth` files to GitHub if they exceed 100MB; use Hugging Face or a public Google Drive link.
* [ ] **Technical Report PDF:** Exported and under the hackathon file size limit.
* [ ] **Demo Video:** Uploaded to YouTube as 'Unlisted', link included in the submission form.
* [ ] **Code Cleansed:** Remove any hardcoded local paths (e.g., `C:/Users/...` or `/content/drive/...`) and replace with relative paths.

---

### 🔭 9. POST-HACKATHON STRATEGY

* **Open Source Release:** Package the inference module as a standalone PyPi library (`pip install infranovatir`) for the remote sensing community.
* **Academic Publication:** If the final SSIM breaks 0.70 on Landsat 9 data, draft a short paper focusing on the downstream YOLO accuracy improvements for the IEEE IGARSS conference or CVPR's EarthVision workshop.