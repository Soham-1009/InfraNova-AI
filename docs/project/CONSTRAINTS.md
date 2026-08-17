# InfraNova AI — System Constraints

This document details hard boundaries and technical constraints that developers must operate within when modifying the codebase.

## 1. Hardware Constraints

- **VRAM Limitations**: Training at 128x128 with a batch size of 256 requires ~28GB of VRAM (comfortably fits across 2x 16GB Tesla T4 GPUs on Kaggle). Increasing `image_size` to 256x256 will OOM (Out of Memory) unless `batch_size` is dropped significantly.
- **Kaggle Execution Limits**: Kaggle notebook sessions are killed after a maximum of 12 hours. The training loop must save resumable checkpoints frequently (currently every epoch) to ensure progress isn't lost during preemption.

## 2. Dataset Constraints

- **Spatial Geometry**: The `Landsat9Dataset` previously enforced strict shape checks at runtime. It has now been updated to issue a warning and dynamically resize patches via `cv2.resize` if they do not match the `image_size` config, though regenerating from Earth Engine is still recommended for maximum performance.
- **Region-Based Splitting**: You cannot randomly shuffle the dataset and split 80/10/10. Satellite imagery overlaps. Random splitting will put patch A in the train set and its neighbor, patch B, in the validation set, causing massive data leakage. The `download_landsat9.py` script enforces strict geographic bounding box splits.
- **Kaggle API Limits**: Uploading the full 15GB raw extraction directory will often crash or timeout the Kaggle API. Always compress and upload only the 4.6GB `data/landsat9/splits/` directory as `splits.zip` to update the Kaggle dataset.

## 3. Mathematical Constraints

- **Activation Ranges**: The Generator's final layer is a `Tanh` activation function. This mathematically restricts its output to the range `[-1.0, 1.0]`. Therefore, the `Landsat9Dataset` *must* normalize all target RGB images to `[-1, 1]`, and the `InferenceEngine` *must* denormalize the output back to `[0, 255]`.
- **Lipschitz Continuity**: The Discriminator uses `SpectralNorm`. Do not add `BatchNorm` or `InstanceNorm` to the Discriminator, as it violates the Lipschitz constraint and breaks the SNGAN mathematical proof for stability.

## 4. Checkpoint Constraints

- **DataParallel Prefixing**: Because the model is trained on multi-GPU Kaggle instances using `nn.DataParallel`, PyTorch automatically prepends `.module.` to all dictionary keys (e.g., `module.down1.conv.weight`). The `checkpoint.py` utility aggressively strips this prefix when loading the model on CPU or single-GPU instances. If you change how the model is wrapped during training, you must update the checkpoint loader.
- **Safe Weights Loading**: PyTorch 2.4+ warns about unsafe deserialization. `checkpoint.py` attempts to load with `weights_only=True` first, but falls back to `weights_only=False` for older checkpoints.

## 5. UI Constraints

- **No Frameworks**: The frontend uses standard React and vanilla CSS variables. Do not install Tailwind CSS, Material UI, or Bootstrap. The UI constraints demand a "glassmorphism" aesthetic built with pure CSS.
- **Vertical Viewport**: The layout is constrained to `100vh`. Do not add scrolling to the main body. If the footer overflows, reduce padding rather than allowing page scroll.
