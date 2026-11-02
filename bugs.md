# InfraNova AI — Bug Ledger & Audit Tracking

**Audit Date:** 2026-10-27  
**Auditor:** Senior Systems & Machine Learning Review (Ponytail Audit)  
**Assumed Production Load:** Single users and research teams uploading satellite scenes (128x128 patches to full scenes) via browser/API, with single-GPU training and evaluation pipelines.

---

## Must Fix (Correctness, Security, Runtime Crashes)

### 1. CORS Wildcard with Credentials Enabled Breaks Browser Requests
- **Location:** [`api/main.py:L63-70`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/api/main.py#L63-L70)
- **What this is:** FastAPI CORS middleware configuration.
- **Problem:** When `INFRANOVA_CORS_ORIGINS` defaults to `*`, `allow_origins=["*"]` is combined with `allow_credentials=True`. W3C CORS specifications strictly forbid wildcard origins when credentials mode is enabled. Modern browsers (Chrome, Firefox, Safari) reject responses with `Access-Control-Allow-Origin: *` when credentials are requested.
- **Fix:** If `CORS_ORIGINS == ["*"]`, set `allow_credentials=False`, or require explicit comma-separated origin URLs when credentials are true.
- **If we skip it:** Frontend web apps hosted on external origins fail with browser CORS errors when attempting to query the API.

---

### 2. Multi-Band GeoTIFF Uploads Truncate Band 11 via PIL in API
- **Location:** [`api/main.py:L158-161`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/api/main.py#L158-L161)
- **What this is:** File parsing in the `POST /colorize` and `POST /predict` endpoints.
- **Problem:** When a user uploads a single `.tif` or `.tiff` file containing both Landsat 9 thermal bands (Band 10 and Band 11, shape `[2, H, W]`), the endpoint reads it with `Image.open(io.BytesIO(raw_bytes))`. Standard PIL reads only the first directory/page (Band 10) and discards Band 11. The model is forced to duplicate Band 10, completely losing split-window atmospheric differential information.
- **Fix:** Use `tifffile.imread(io.BytesIO(raw_bytes))` when suffix is in `{".tif", ".tiff"}` (already a production requirement in `requirements-prod.txt`), falling back to PIL only for standard formats (`.png`, `.jpg`).
- **If we skip it:** Scientific GeoTIFF files with dual thermal bands lose Band 11, degrading colorization fidelity.

---

### 3. Case-Sensitive File Extension Checks Reject `.NPY` Uploads
- **Location:** [`api/main.py:L138,L143`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/api/main.py#L138)
- **What this is:** Dual-band upload parsing for `band10` and `band11` parameters.
- **Problem:** `(band10.filename or "").endswith(".npy")` is case-sensitive. If an operating system or user uploads `scene_B10.NPY` or `scene_B11.TIF`, the check fails and routes the buffer to `Image.open()`, raising `UnidentifiedImageError` (HTTP 400).
- **Fix:** Normalize to lowercase: `(band10.filename or "").lower().endswith(".npy")`.
- **If we skip it:** Users uploading `.NPY` files from Windows or camera pipelines experience unexpected crashes.

---

### 4. Radiometric Augmentations Are a No-Op Under Default Local Normalization
- **Location:** [`src/datasets/landsat9_dataset.py:L220-238,L324-344`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/datasets/landsat9_dataset.py#L220-L238)
- **What this is:** Paired brightness and contrast data augmentations in `Landsat9Dataset`.
- **Problem:** In `__getitem__`, `_apply_augmentation` multiplies `x` and `y` by `factor` ($1.0 \pm 0.1$) on raw arrays. Immediately afterwards, `_normalize_tir_local()` computes `(arr - p_low) / (p_high - p_low)`. Because percentiles scale linearly with scalar multiplication, `factor` cancels out in the numerator and denominator: $\frac{f \cdot x - f \cdot p_{low}}{f \cdot p_{high} - f \cdot p_{low}} = \frac{x - p_{low}}{p_{high} - p_{low}}$. The brightness augmentation has zero effect on training data.
- **Fix:** Apply radiometric augmentations after normalization in `[-1.0, 1.0]` space, or apply them to unnormalized data when using `normalization="global"`.
- **If we skip it:** Generator misses intended photometric invariance during training.

---

### 5. Unbounded Tensor Inputs After Noise Augmentation
- **Location:** [`src/datasets/landsat9_dataset.py:L234-238`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/datasets/landsat9_dataset.py#L234-L238)
- **What this is:** Gaussian noise injection on TIR inputs during dataset augmentation.
- **Problem:** `x = x + noise` adds sensor noise ($\sigma = 0.02$) without clipping. Augmented values exceed the strict $[-1.0, 1.0]$ boundary expected by model reflection padding and Tanh bounds.
- **Fix:** Apply `np.clip(x, -1.0, 1.0)` following radiometric noise additions.
- **If we skip it:** Extreme pixel outliers enter downsampling layers, causing gradient spikes.

---

### 6. Grid Coordinate Indexing Bug for Small Rasters in Tiled Inference
- **Location:** [`src/inference/tiled_inference.py:L198-204`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/inference/tiled_inference.py#L198-L204)
- **What this is:** Calculation of sliding window grid offsets for whole-scene inference.
- **Problem:** If a raster has dimension smaller than `tile_size` (e.g. $100 \times 100$ with $128 \times 128$ tile size), `range(0, max(1, H - 128 + 1), stride)` yields `[0]`. Then `if y_starts[-1] + self.tile_size < H` evaluates `128 < 100` (False). If $H < 128$, `y_starts` contains `[0]`, slicing `[2, 100, 100]` without adjusting coordinates, causing dimension mismatch in batch stacking.
- **Fix:** Explicitly guard at function start: if $H < \text{tile\_size}$ or $W < \text{tile\_size}$, pad the scene to at least $(\text{tile\_size}, \text{tile\_size})$ before calculating grid starts.
- **If we skip it:** Sub-patch test rasters trigger tensor shape mismatch exceptions during batch stacking.

---

## Should Fix (Performance, Memory Leaks, Fragile Abstractions)

### 7. VGG Perceptual Loss Retains Backward Graph on Ground-Truth Images
- **Location:** [`src/training/losses.py:L298-305`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/training/losses.py#L298-L305)
- **What this is:** Feature map extraction for perceptual loss calculation.
- **Problem:** In `VGGPerceptualLoss.forward()`, `current_real = block(current_real)` is computed without `.detach()`. While VGG parameters have `requires_grad=False`, computing intermediate activations from `real_rgb` builds an active computation graph across 4 VGG blocks, keeping redundant activation tensors in GPU memory.
- **Fix:** Wrap target pass in `with torch.no_grad():` or call `current_real = block(current_real).detach()`.
- **If we skip it:** Consumes 200–400 MB of unnecessary GPU VRAM per training step.

---

### 8. Module State Mutation During Forward Pass in SSIMLoss
- **Location:** [`src/training/losses.py:L347-352`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/training/losses.py#L347-L352)
- **What this is:** Differentiable SSIM loss calculation.
- **Problem:** In `_ssim()`, when channel or device changes, `self.window = window` assigns a new tensor directly to module attributes during the forward pass. Mutating registered buffers or module attributes in `forward()` breaks TorchScript tracing, `torch.compile`, and causes DDP buffer broadcast mismatches.
- **Fix:** Return the window as a local variable from `_create_window` if device/channel mismatches, or create windows with an `@lru_cache` helper keyed by `(window_size, sigma, channel, device, dtype)`.
- **If we skip it:** Graph capture, JIT compilation, and distributed training fail with state mutation errors.

---

### 9. Shared GradScaler Step Conflict in GAN Training
- **Location:** [`src/training/trainer.py:L320-333,L374-386`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/training/trainer.py#L320-L333)
- **What this is:** Mixed precision gradient scaling for Discriminator and Generator.
- **Problem:** Both `self.optimizer_d` and `self.optimizer_g` are stepped with the same `self.scaler` instance in sequence, calling `self.scaler.update()` twice per batch. If `optimizer_d` encounters non-finite gradients, the scaler reduces the loss scale prematurely before `optimizer_g` runs.
- **Fix:** Instantiate two distinct scalers (`self.scaler_d` and `self.scaler_g`), or perform `self.scaler.step()` for both optimizers and call `self.scaler.update()` exactly once per batch.
- **If we skip it:** Unstable gradient scaling in mixed-precision training causes unpredictable loss spikes.

---

### 10. Duplicate Inference Engine Implementations
- **Location:** [`demo/inference.py:L22-230`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/demo/inference.py#L22-L230) vs [`src/inference/landsat_inference.py:L34-324`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/src/inference/landsat_inference.py#L34-L324)
- **What this is:** Model inference abstractions.
- **Problem:** Both modules implement parallel classes (`InferenceEngine` and `LandsatColorizationInference`) that perform model loading, array conversion, percentile stretching, and TTA. `demo/inference.py` uses 4 TTA transforms (rot180 included) while `landsat_inference.py` uses 3. Two implementations must be maintained in sync.
- **Fix:** Consolidate into `src/inference/landsat_inference.py` and have `demo/inference.py` import and wrap it.
- **If we skip it:** Bug fixes and performance enhancements applied to one engine are missed in the other.

---

### 11. Browser Memory Leak from Unrevoked Blob URLs
- **Location:** [`web/src/App.jsx:L54-75`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/web/src/App.jsx#L54-L75)
- **What this is:** Client-side preview and inference result image tracking.
- **Problem:** Every time a user colorizes an image or applies CLAHE, `URL.createObjectURL(blob)` appends to `blobUrlsRef.current`. `cleanupBlobUrls` is only triggered on complete component unmount, retaining uncompressed raster textures in browser memory across continuous interactive sessions.
- **Fix:** Revoke the previous output/display URL before storing the new one: `if (outputImage) URL.revokeObjectURL(outputImage)`.
- **If we skip it:** Extended browser sessions accumulate hundreds of megabytes of image textures.

---

### 12. Hardcoded Absolute Sample Paths Break Subpath Hosting
- **Location:** [`web/src/App.jsx:L11-28,L85`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/web/src/App.jsx#L11-L28)
- **What this is:** Static preset image paths in frontend client.
- **Problem:** Preset assets are referenced as `/samples/accra_thermal.png`. When deployed under a subpath (e.g. GitHub Pages at `/InfraNova-AI/`), absolute root paths resolve to `https://<domain>/samples/...` instead of `https://<domain>/InfraNova-AI/samples/...`, causing HTTP 404s.
- **Fix:** Prepend `import.meta.env.BASE_URL` to public asset paths: `${import.meta.env.BASE_URL}samples/...`.
- **If we skip it:** Preset sample cards show broken image icons when viewed on GitHub Pages.

---

## Nice to Have (Speed & Code Hygiene)

### 13. Global Flattened SSIM in Fallback Evaluation
- **Location:** [`scripts/evaluation/evaluate.py:L43-62`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/scripts/evaluation/evaluate.py#L43-L62)
- **What this is:** Fallback SSIM calculation when `torchmetrics` is unavailable.
- **Problem:** Computes global scalar variance across the entire image instead of localized sliding-window patches ($11 \times 11$). Results overestimate SSIM compared to standard windowed implementations.
- **Fix:** Use OpenCV or a lightweight local windowed filter if `torchmetrics` is missing.

### 14. StreamingResponse Without Content-Length for Fixed Buffer Outputs
- **Location:** [`api/main.py:L173-180`](file:///c:/Users/soham/Desktop/Soham/InfraNova-AI/api/main.py#L173-L180)
- **What this is:** HTTP response generation for synthesized PNG images.
- **Problem:** `StreamingResponse(buf)` uses `Transfer-Encoding: chunked`. Standard `Response(content=buf.getvalue(), media_type="image/png")` is more lightweight and provides `Content-Length`, improving caching performance in CDNs and reverse proxies.
