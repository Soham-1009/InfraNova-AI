# InfraNova AI — Inference Architecture

This document explains how the trained model processes novel data in production.

## 1. Engine Selection

There are two inference engines in the codebase:
1. `src/inference/landsat_inference.py` (`LandsatColorizationInference`): Designed for batch processing large sets of GeoTIFFs. Handles geospatial metadata and expects `image_size` as a multiple of 256.
2. `demo/inference.py` (`InferenceEngine`): Designed specifically for the Web API. Optimized for PIL Images, handles base64 conversions gracefully, and correctly supports the model's native 128×128 training resolution.

**The production FastAPI backend exclusively uses `demo/inference.py`.**

## 2. Preprocessing

Before passing an image to the model, it must match the mathematical distribution of the training data.

1. **Grayscale Conversion**: If the user uploads an RGB image, it is converted to luminance via `0.299*R + 0.587*G + 0.114*B`. 16-bit TIFFs preserve their native precision.
2. **Percentile Stretching**: Thermal data lacks a fixed "black" or "white" point (temperatures vary by season and location). The engine computes the 1st (p1) and 99th (p99) percentiles of the specific input image. The image is clamped to this range and scaled from `0.0` to `1.0`.
3. **Resizing**: Resized via Bicubic interpolation to 128×128.
4. **Normalization**: Scaled to `[-1, 1]` to match the GAN's `Tanh` activation range.
5. **Tensor Conversion**: Cast to `torch.float32` of shape `[1, 1, 128, 128]`.

## 3. Test-Time Augmentation (TTA)

To improve inference quality, the engine uses TTA. By default, `use_tta=True` is passed by the API.

1. The preprocessed tensor `ir` is fed into the generator 4 times:
   - Pass 1: Original `ir`
   - Pass 2: Horizontal flip of `ir`
   - Pass 3: Vertical flip of `ir`
   - Pass 4: 180-degree rotation of `ir` (H+V flip)
2. The model generates 4 corresponding RGB outputs.
3. The geometric transforms are inverted on the 4 outputs (e.g., the output of Pass 2 is horizontally flipped back to normal).
4. The 4 output tensors are averaged together `torch.mean(dim=0)`.

**Why this works**: Convolutional networks are not perfectly shift/rotation invariant. TTA smoothes out deterministic artifacts (like a tendency to draw roads curving slightly to the left) and produces a higher quality, more consistent colorization.

## 4. Postprocessing

1. **Denormalization**: The averaged tensor is scaled from `[-1, 1]` back to `[0, 1]`.
2. **Quantization**: Multiplied by 255.0 and rounded to `np.uint8`.
3. **Array Conversion**: Converted from PyTorch's `CHW` format to PIL's `HWC` format.
4. **Output**: Returned as a PIL Image to the API.

*Note: CLAHE contrast enhancement (`demo/utils.py:enhance_output`) is available but not currently utilized by the API to preserve the model's raw output fidelity.*
