# InfraNova AI — System Constraints

This document details hard boundaries and technical constraints that developers must operate within when modifying the codebase.

## 1. Hardware Constraints

- **VRAM Limitations**: Training at 128x128 with a batch size of 128 requires ~14GB of VRAM (comfortably fits on a 16GB Tesla T4). Increasing `image_size` to 256x256 will OOM (Out of Memory) unless `batch_size` is dropped to ~32.
- **Kaggle Execution Limits**: Kaggle notebook sessions are killed after a maximum of 12 hours. The training loop must save resumable checkpoints frequently (currently every epoch) to ensure progress isn't lost during preemption.

## 2. Dataset Constraints

- **Strict Spatial Geometry**: The `Landsat9Dataset` enforces a strict shape check at runtime (`tir_100m.shape[-2:] == (image_size, image_size)`). It **does not** dynamically crop or resize files on disk that don't match the config. If you change `image_size` in the config, you *must* regenerate the entire dataset via Earth Engine.
- **Region-Based Splitting**: You cannot randomly shuffle the dataset and split 80/10/10. Satellite imagery overlaps. Random splitting will put patch A in the train set and its neighbor, patch B, in the validation set, causing massive data leakage. The `download_landsat9.py` script enforces strict geographic bounding box splits.

## 3. Mathematical Constraints

- **Activation Ranges**: The Generator's final layer is a `Tanh` activation function. This mathematically restricts its output to the range `[-1.0, 1.0]`. Therefore, the `Landsat9Dataset` *must* normalize all target RGB images to `[-1, 1]`, and the `InferenceEngine` *must* denormalize the output back to `[0, 255]`.
- **Lipschitz Continuity**: The Discriminator uses `SpectralNorm`. Do not add `BatchNorm` or `InstanceNorm` to the Discriminator, as it violates the Lipschitz constraint and breaks the SNGAN mathematical proof for stability.

## 4. Checkpoint Constraints

- **DataParallel Prefixing**: Because the model is trained on multi-GPU Kaggle instances using `nn.DataParallel`, PyTorch automatically prepends `.module.` to all dictionary keys (e.g., `module.down1.conv.weight`). The `checkpoint.py` utility aggressively strips this prefix when loading the model on CPU or single-GPU instances. If you change how the model is wrapped during training, you must update the checkpoint loader.
- **Safe Weights Loading**: PyTorch 2.4+ warns about unsafe deserialization. `checkpoint.py` attempts to load with `weights_only=True` first, but falls back to `weights_only=False` for older checkpoints.

## 5. UI Constraints

- **No Frameworks**: The frontend uses standard React and vanilla CSS variables. Do not install Tailwind CSS, Material UI, or Bootstrap. The UI constraints demand a "glassmorphism" aesthetic built with pure CSS.
- **Vertical Viewport**: The layout is constrained to `100vh`. Do not add scrolling to the main body. If the footer overflows, reduce padding rather than allowing page scroll.
