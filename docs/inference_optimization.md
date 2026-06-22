## 1. Performance Estimates

| Hardware | Inference Time | Notes |
|----------|---------------|-------|
| GPU T4 (fp16) | ~10ms | Tensor cores, batch size 1 |
| GPU T4 (fp32) | ~20ms | No half precision |
| CPU (PyTorch) | 2-3 seconds | Modern laptop |
| CPU (ONNX quantized) | 0.5-1s | 2-4x speedup |

## 2. Memory Footprint

- Parameters (fp32): 228 MB
- Parameters (fp16): 114 MB
- Activations (256x256, fp32): 800 MB
- Total RAM needed: ~1 GB (fp32) or 500 MB (fp16)

## 3. Optimization Strategy

### For Laptop (CPU)
- Use PyTorch with @st.cache_resource for model loading
- Disable gradients with torch.no_grad()
- Accept 2-3 second inference time
- Show progress spinner during inference

### For GPU Deployment
- Convert to fp16: model.half()
- Use 2x speedup with Tensor Cores

### For Production
- Export to ONNX + quantize to int8
- Deploy with ONNX Runtime
- Achieve sub-second CPU inference

---

## 4. Input Validation Rules

### Acceptable Formats
- PNG, JPG, TIFF
- Max file size: 20 MB
- Max resolution: 4096x4096

### Preprocessing
1. If RGB uploaded, extract red channel only as IR
2. Resize maintaining aspect ratio
3. Pad to square with zeros
4. Resize to 256x256 with Lanczos interpolation
5. Normalize to [-1, 1]

### Auto-reject
- Files > 20 MB
- Non-image MIME types
- Truncated/corrupt images
- Resolution > 4096px

---

## 5. Output Enhancement

### Test-Time Augmentation (TTA)
Run inference on 4 augmented versions, average results:
- Original
- Horizontal flip
- Vertical flip
- 90 degree rotation

Improves SSIM by 0.01-0.02. Costs 4x inference time.

### Post-processing
- CLAHE on luminance channel for contrast
- Convert from [-1, 1] to [0, 255]
- Optional sharpening (unsharp mask)

### Edge Cases
- Mostly black input (mean < 0.01): Return black with warning
- Mostly white input (mean > 0.99): Return white with warning
- NaN/inf in output: Return grayscale IR with error

---

## 6. Confidence Score

Use discriminator output as quality metric:
- > 0.7: High quality
- 0.4-0.7: Acceptable
- < 0.4: Low confidence warning

---

## 7. Demo Metrics to Display

| Metric | Always Show? | Source |
|--------|--------------|--------|
| Inference time | Yes | Timer around model call |
| Output dimensions | Yes | Output tensor shape |
| Confidence score | Yes | Discriminator output |
| PSNR | Only if ground truth | User uploads target |
| SSIM | Only if ground truth | User uploads target |

---

## 8. Security & Robustness

### File Upload Security
- Validate MIME types: image/png, image/jpeg, image/tiff
- Use PIL.verify() for truncated files
- Max 20 MB file size

### Memory Protection
- Resize images > 2048px before inference
- Use torch.no_grad() context
- Call torch.cuda.empty_cache() after GPU inference
- 30 second timeout on CPU, 5 seconds on GPU

### Rate Limiting (for production)
- 1 request per 2 seconds per IP
- Queue max 10 requests

---

## 9. Scalability

### Single GPU Capacity
- T4 can serve 50 requests/second
- Handles 10 concurrent users comfortably

### CPU Demo Limits
- 1 request every 2 seconds sequentially
- Use async workers for concurrent users

### Cache Strategy
- Hash input image
- LRU cache for repeated inputs
- Cache size: 32 most recent

---

## Final Recommendations

**For Hackathon Demo (Laptop):**
- Use PyTorch on CPU
- Single image inference (2-3s)
- Show spinner during processing
- Pre-render sample card outputs

**For Production Deployment:**
- GPU with fp16 + TTA for highest quality
- CPU with ONNX quantized for cost-effective deployment
- Always show confidence score
