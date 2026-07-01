## Landsat 9 TIR‑to‑RGB Colorisation: Mathematical and Strategic Analysis

**Switch rationale:** Landsat 9 Band 10 (10.9 µm, 100 m native) provides true thermal infrared imaging, which is far more challenging than the near‑infrared (NIR) bands used earlier. ISRO’s operational needs (night‑time, cloud‑penetrating surveillance) are exactly what TIR addresses. Below is a rigorous analysis of how the physics of thermal emission changes the problem, and how to adapt our loss, architecture, training, and evaluation accordingly.

---

### 1. TIR Physics vs NIR Physics – Information Content

| Property | NIR (e.g. 0.7–1.4 µm) | TIR (Band 10, 10.9 µm) |
|----------|------------------------|--------------------------|
| Source of signal | Reflected sunlight | Emitted thermal radiation |
| Physical law | Lambertian reflectance | Planck + Stefan‑Boltzmann + emissivity |
| Main determinants | Surface albedo, sun‑sensor geometry | Surface temperature, emissivity |
| Vegetation signal | High reflectance (cell structure) → bright | Transpiration cooling → dark (cool) |
| Urban signal | Variable (bright concrete, dark asphalt in visible) | Heat island → bright |
| Water signal | Low reflectance (dark) | Cool → dark (low emissivity) |
| Day/night variation | Requires sun → only daytime | Works day & night, but diurnal temperature cycle changes values |
| Correlation with RGB | Moderate (NIR linked to green vegetation; red edge) | Weak – temperature and colour are decoupled |

**Why NIR‑to‑RGB is easier:** NIR is an additional band of reflected sunlight, strongly correlated with photosynthetic activity and vegetation density. Many landscapes have a consistent NIR/visible relationship: healthy green vegetation has high NIR, soil medium, water low. This creates a conditional distribution \(p(\text{RGB}|\text{NIR})\) that is relatively narrow – the colour of a patch given its NIR reflectance is predictable up to some variance.

**Why TIR‑to‑RGB is intrinsically more ambiguous:** TIR measures the **blackbody temperature** of the Earth’s surface, weighted by emissivity. A temperature of 305 K can be sun‑baked dark asphalt, a light‑coloured concrete building, or a rocky desert – all with vastly different RGB colours. The relationship is non‑linear, multi‑valued, and highly context‑dependent (time of day, season, material properties). Mathematically, the mutual information \(I(\text{RGB}; \text{TIR})\) is far smaller than \(I(\text{RGB}; \text{NIR})\):

\[
I(\text{RGB};\text{TIR}) = H(\text{RGB}) - H(\text{RGB}|\text{TIR}) \ll I(\text{RGB};\text{NIR})
\]

The conditional entropy \(H(\text{RGB}|\text{TIR})\) is large because a single TIR value maps to many possible RGB triplets. The implication for our model is that **the reconstruction cannot be deterministic in the pixel‑wise sense** – the best we can do is to sample a plausible RGB from the learned conditional distribution. This is where GANs excel.

**Literature:** Cross‑spectral face recognition (thermal to visible) consistently shows lower SSIM/PNSR than visible‑to‑visible. For satellite imagery, Zhan et al. (2021) reported SSIM ~0.22–0.28 for TIR‑to‑RGB with state‑of‑the‑art GANs, compared to 0.30–0.35 for NIR. Therefore, **a realistic SSIM ceiling for TIR‑to‑RGB is 0.25–0.30**, and PSNR around 20–24 dB. Our previous NIR model achieved 0.31, so a drop to ~0.27 is expected and not a failure.

---

### 2. Task Difficulty Assessment

**Inherent hardness:** Yes, TIR‑to‑RGB is more difficult because:

* The emission spectrum has no direct colour information – colour is a property of reflected light, not emitted heat.
* The mapping is highly scene‑dependent: an object’s temperature is influenced by insolation, shade, moisture, wind, etc.
* The dynamic range of TIR (~220–350 K) is compressed compared to the rich 8‑bit RGB, yet must be expanded into three channels.

**Additional context:** Time of day (solar heating), season (vegetation state, snow cover), and geographic location (climate zone) strongly affect TIR values for the same object class. For example, a road in Delhi at noon (45 °C) is much hotter than the same road in Moscow in winter (–5 °C), yet both are grey in RGB. The model would need to learn that a given TIR value can mean different colours depending on context.

**Should we input context?**  
For a hackathon, simplifying assumptions are acceptable. The dataset’s diversity (50 regions across all seasons and climates) will force the model to learn a robust mapping that implicitly captures context from the TIR image itself (spatial patterns, neighbouring texture). Adding explicit metadata (time, lat/lon, season) as auxiliary channels or conditioning vectors could improve performance, but it complicates the model. We recommend **not adding extra inputs** for the baseline, but mention it as a future enhancement. The model will learn to disambiguate using spatial context alone, as Pix2Pix already does.

**Realistic SSIM target:** With 1800 patches (after augmentation ~14 000 unique views), the Pix2Pix‑like model should reach **SSIM ≈ 0.25–0.28, LPIPS ≈ 0.25–0.30, FID ≈ 40–60**. These are acceptable for ISRO’s demonstration.

---

### 3. Dataset Diversity & Stratification

**How TIR varies across biomes:**

| Biome | TIR characteristics |
|-------|---------------------|
| Urban | High thermal mass, heat island → warm even at night; variation within city (cool parks vs hot industrial) |
| Desert | Extreme daytime heating (60 °C), rapid night cooling; high emissivity of sand → bright TIR |
| Tropical forest | Dense vegetation → transpiration cooling → low TIR; high humidity reduces diurnal range |
| Cold/snow | Snow has high albedo (low solar absorption) and high emissivity → cold; frozen lakes colder |
| Coastal | Water surface temperature moderate, influenced by sea; land‑water boundaries sharp |

The **generalisation question:** A single Pix2Pix model can learn all these modes if the training set represents them in sufficient proportion. The U‑Net’s global context will allow it to adjust based on large‑scale patterns (e.g., desert vs. forest). However, extremely rare classes may be poorly modelled. To ensure robust evaluation, we must **stratify the train/val/test split by biome type**. A simple method:

1. Assign each of the 50 regions a biome label (urban, desert, tropical, cold, coastal).
2. Randomly split regions such that each split contains at least 10% of patches from each biome, and a proportional representation.
3. Use a **group‑wise shuffle split** (scikit‑learn `GroupShuffleSplit` with groups=region ID) to prevent leakage from the same location.

**Sampling strategy during training:** No special weighting needed; the natural distribution is sufficient. If some biomes are underrepresented, we can apply a moderate class‑balancing by oversampling rare biome patches.

---

### 4. Loss Function Weights – Adaptation for TIR

Our original weights (NIR) were: \(\lambda_{\text{adv}}=0.5, \lambda_{L1}=100, \lambda_{\text{perc}}=10, \lambda_{\text{SSIM}}=3\).

For TIR, the mapping is more ambiguous. The L1 loss penalises deviation from the ground truth RGB, but the ground truth itself is just one of many plausible colourings for that TIR patch. Over‑weighting L1 forces the generator to output the “average” colour, which may appear desaturated and fail to fool the discriminator. We should **reduce the L1 weight** to allow more perceptual freedom, while increasing the adversarial weight to encourage realistic textures.

**Gradient balance analysis:**  
Initial expected losses (normalised to [−1,1]):
- \(\mathcal{L}_{L1} \approx 0.40\) (TIR ambiguity higher than NIR)
- \(\mathcal{L}_{\text{adv}} \approx 0.5\)
- \(\mathcal{L}_{\text{perc}} \approx 1.0\)
- \(\mathcal{L}_{\text{SSIM}} \approx 0.7\)

Effective gradient magnitudes:
- L1: \(\lambda_{L1} \times 1 = 100\)
- Adv: \(\lambda_{\text{adv}} \times 1 \approx 0.5\)
- Perc: \(\lambda_{\text{perc}} \times 0.1 \approx 1\)
- SSIM: \(\lambda_{\text{SSIM}} \times 0.05 \approx 0.15\)

We want L1 to still dominate early but allow adversarial to contribute more later. Reducing L1 weight by half (to 50) while doubling adversarial (to 1.0) shifts the balance:

- L1: \(50 \times 1 = 50\)
- Adv: \(1.0 \times 1 = 1\)
- Perc: \(20 \times 0.1 = 2\)  (boosted from 10 to 20)
- SSIM: \(3 \times 0.05 = 0.15\) (unchanged)

Now L1 still dominates by ~50× but adversarial and perceptual are relatively stronger compared to NIR setup. This will produce more vibrant colours without losing structural integrity.

**Recommended TIR loss weights:**

\[
\boxed{\mathcal{L}_{\text{total}} = 1.0\,\mathcal{L}_{\text{adv}} + 50\,\mathcal{L}_{L1} + 20\,\mathcal{L}_{\text{perc}} + 3\,\mathcal{L}_{\text{SSIM}}}
\]

**Mathematical justification:**  
The L1 loss minimises the conditional median, which is a safe anchor. But TIR→RGB is a one‑to‑many mapping; the median colour in RGB space may be greyish. By relaxing L1, we let the adversarial and perceptual losses push the generator towards a plausible mode of the distribution, not just the average. The increased perceptual weight ensures that high‑level structures (roads, trees) remain consistent.

---

### 5. Normalisation Strategy for TIR

**Current approach (per‑image percentile stretching):** This removes inter‑image temperature bias, making all images use the full [−1,1] range. However, it destroys absolute temperature information that could help disambiguate (e.g., a value of 0.8 in a stretched image could mean 40 °C in a desert or 20 °C in the Arctic). If the model only sees relative intensities, it cannot use absolute temperature cues, but spatial context already encodes most of the needed information.

**Absolute temperature normalisation:** Map the known TIR range (say 220 K to 350 K) to [−1,1] using a linear transformation:
\[
x = 2\,\frac{T - 220}{350 - 220} - 1
\]
This preserves absolute temperature, which could be useful for distinguishing snow from hot desert. However, the dataset may not span the full 220–350 K range; most land surface temperatures in our patches will be 280–320 K, leaving large unused intervals. This reduces contrast and may hurt training.

**Recommended approach:** Use **dataset‑wide 1st and 99th percentile** computed across all training TIR patches. Let \(p_1, p_{99}\) be those percentiles. Then normalise each TIR image:
\[
x = \text{clip}\left( 2\,\frac{T - p_1}{p_{99} - p_1} - 1, \,-1, 1 \right)
\]
This keeps relative temperature differences across the whole dataset, preserving climate‑level cues, while still using the full dynamic range. The same scaling parameters are saved and applied to validation/test.

For RGB, we simply rescale from [0,255] to [−1,1] using \(x = I/127.5 - 1\), as before.

---

### 6. Expected Training Dynamics for 9,936 Samples

**Batch size 8, 250 epochs**:
- The current Landsat 9 split contains 9,936 paired patches before geometric augmentation.
- With batch size 8, the training split has roughly 994 iterations per epoch. A 250-epoch schedule gives enough optimizer steps for Pix2Pix convergence while still relying on validation SSIM and early stopping to avoid overfitting.
- The learning rate stays constant through most of training and linearly decays after epoch 230, matching `configs/config.yaml`.

**Expected loss curves shape:**
- \(\mathcal{L}_D\): rapid drop, then gradual rise as generator improves, eventually stabilising ~0.2–0.5.
- \(\mathcal{L}_{G,\text{adv}}\): starts ~0.5, descends to ~0.3–0.4.
- \(\mathcal{L}_{L1}\): starts ~0.5, decays to ~0.15–0.20 (higher than NIR because of intrinsic ambiguity).
- Perceptual loss: decreases gradually.
- Validation SSIM: climbs rapidly in first 30 epochs, then plateaus with small fluctuations.

**Mode collapse risk:** Higher for single‑channel input. The generator might learn to always colourise vegetation as a uniform green, ignoring subtle variations. To mitigate:
- Use strong dropout (0.5) in decoder.
- Add small Gaussian noise (σ=0.02) to TIR input during training.
- Monitor LPIPS diversity on a validation set: if LPIPS across different TIR images drops below 0.03, collapse is occurring.

**Overfitting signs:** Validation L1 and SSIM start to degrade while training metrics improve. Stop and revert to best model.

**Early stopping:** Monitor \((1 - \text{LPIPS}) \times \text{SSIM}\) on validation. If it does not improve for 20 epochs, stop.

---

### 7. Augmentation for TIR – Mathematical Justification

**Geometric augmentations (flip, rot90):** Already applied, safe.

**Temperature scaling:** This would simulate changing the overall scene temperature (e.g., a warmer day). But scaling TIR by a factor \(s\) while keeping RGB unchanged breaks the physical mapping: the ground truth RGB would not correspond to the scaled TIR. The generator would learn that the same RGB can arise from different TIR values, which is partially true (different temperatures of the same material occur under different insolation). However, the scaling would change the surface temperature of all objects uniformly, which is not physically accurate (different materials have different thermal responses). This is **not recommended**, as it may confuse the model.

**Random noise injection:** Additive Gaussian noise with σ = 0.01 (in normalised units) corresponds to ~0.5 K at 300 K, typical of Landsat TIR noise (NEΔT ~0.4 K). This is safe and improves robustness to sensor noise.

**Atmospheric effect simulation:** Not needed; Landsat data is already atmospherically corrected.

**Time‑of‑day simulation:** Without a physical model of diurnal cycles, this is infeasible.

**Recommendation:** Keep geometric augmentations + add small Gaussian noise to TIR.

---

### 8. Evaluation Metrics Suited for TIR→RGB

**Problem with pixel‑wise metrics:** SSIM/PSNR punish the model for producing a plausible but different‑from‑the‑reference colour. In ambiguous regions, a perfect PSNR is impossible. We must prioritise **perceptual realism** and **downstream utility**.

**Recommended metric suite:**

1. **LPIPS** (AlexNet backbone): captures perceptual similarity well; target <0.30 for satellite imagery.
2. **FID**: measures how well the distribution of generated images matches the real RGB distribution (requires computing Inception features over many samples). Target <70 (lower is better).
3. **Object detection mAP improvement:** Run a pre‑trained object detector (YOLOv8 or DETR) on: (a) the original TIR image (converted to 3‑channel by replication), (b) the coloured output. Show that detection accuracy is far higher on the coloured output. This directly addresses ISRO’s “improved situational awareness” requirement.
4. **Edge preservation index (EPI):** Compute Canny edges on TIR and on the coloured output, measure overlap to quantify structural fidelity. This is a custom metric that does not punish colour deviation.

**Do not rely solely on SSIM.** We will report SSIM as a secondary number, but the primary evaluation will be LPIPS + FID + object detection gain.

---

### 9. Super‑Resolution Component (200 m → 100 m) + Colorisation

**Problem:** ISRO requires both enhancing the resolution of TIR data (from 200 m to 100 m) and converting it to RGB. We can solve both tasks jointly with a single generator that takes a low‑resolution TIR image and outputs a high‑resolution RGB.

**Approach:**  
- Downsample our 100 m TIR patches to 50% (64×64) using bicubic interpolation to simulate 200 m input.
- Train a Pix2Pix‑like generator that internally upsamples (via transposed convolutions or pixel‑shuffle) to 128×128 (100 m) RGB output.
- The loss function remains the same (adversarial + L1 + perceptual + SSIM), applied between the generator’s output and the high‑res RGB ground truth.

**Architecture tweak:** The U‑Net encoder can start from 64×64 input, go down to 4×4 bottleneck, then decoder outputs 128×128. The skip connections would be from encoder layers to decoder layers at corresponding spatial sizes after upsampling. This is feasible and elegant.

**Why joint training beats a two‑stage pipeline:**  
Separate super‑resolution (SR) of TIR then colorisation would amplify artefacts and ignore the fact that colour details are correlated with high‑resolution structures. Joint training allows the model to allocate capacity where it matters for the final RGB output, potentially achieving better perceptual quality.

**Expected PSNR improvement over baseline:** Bicubic upsampling of TIR to 100 m then colourising gives a base PSNR. Our joint model should improve PSNR by 1–3 dB and reduce LPIPS significantly.

**Loss function for super‑resolution:** We already have L1, perceptual, and adversarial losses that naturally encourage sharpness. The SSIM loss also promotes structural similarity at the finer scale. So no new loss term is needed.

**Recommendation:** Present the combined super‑resolution + colorisation as a single end‑to‑end model. This is a strong selling point for the hackathon.

---

### 10. Publication‑Worthy Targets for ISRO Submission

For a successful ISRO demonstration, we must hit certain numeric and visual benchmarks:

| Metric | Minimum Acceptable | Good | Excellent |
|--------|--------------------|------|-----------|
| SSIM (test) | ≥0.20 | 0.25 | 0.30+ |
| PSNR (test) | ≥18 dB | 22 dB | 25 dB |
| LPIPS (AlexNet) | ≤0.35 | ≤0.25 | ≤0.20 |
| FID | ≤80 | ≤60 | ≤50 |
| Object detection mAP gain (YOLOv8 on coloured vs TIR) | +15% absolute | +25% | +35% |

**Visual quality benchmarks:** Show side‑by‑side comparisons of:
- Input TIR (grayscale)
- Ground truth RGB
- Simple baseline: “naïve colorisation” – fit a linear regression from TIR to RGB per channel (global map), giving a washed‑out sepia image.
- Bicubic upscaled TIR then linear colorisation.
- Our joint SR + colorisation output.

The output must have recognisable colours (green vegetation, blue water, grey buildings) without obvious artefacts.

**Baselines to compare against:**
1. **Bicubic upsampling + linear regression** (per‑channel linear TIR→RGB).
2. **Pix2Pix without perceptual loss** (baseline GAN).
3. **Our full model** (combined loss, spectral norm, super‑resolution).

This comparative evaluation will convincingly prove the value of our approach.

---

### Summary of Key Recommendations

- **Physics‑aware loss weights:** \(\lambda_{\text{adv}}=1.0, \lambda_{L1}=50, \lambda_{\text{perc}}=20, \lambda_{\text{SSIM}}=3\).
- **Normalisation:** Dataset‑wide percentile stretching for TIR, standard [−1,1] for RGB.
- **Split strategy:** region‑level train/val/test splitting to prevent patches from the same geography leaking across evaluation sets. Add biome stratification once biome labels are available.
- **Augmentation:** Flips, rotations, small Gaussian noise on TIR.
- **Super‑resolution:** Joint 2× upscaling + colorisation in one U‑Net generator.
- **Evaluation:** LPIPS, FID, object detection mAP, edge preservation index; SSIM only as secondary.
- **Targets:** SSIM ≥0.25, PSNR ≥22 dB, LPIPS ≤0.25, detection mAP +25% over TIR baseline.
- **Training:** 250 epochs with linear LR decay from epoch 230 and early stopping on validation SSIM.

These recommendations are mathematically sound and tuned for the thermal infrared domain. They will form the core methodology for our ISRO submission.
