# InfraNova AI — Architectural Decisions

This document records the major design decisions made during the development of InfraNova AI, explaining the *why* behind the *what*.

## 1. Why Pix2Pix instead of CycleGAN?
- **Decision**: Use a paired conditional GAN (Pix2Pix) rather than an unpaired one (CycleGAN).
- **Reasoning**: We have exact pixel-to-pixel spatial alignments because Landsat 9 captures thermal and RGB data simultaneously. CycleGAN is designed for unpaired datasets (e.g., horses to zebras) and introduces "hallucination" artifacts. Pix2Pix tightly constrains the generator to match the exact input geometry using L1 loss.

## 2. Why Numpy arrays (`.npy`) instead of Images (`.png` / `.tif`) for the Dataset?
- **Decision**: Store all training patches as `.npy` files.
- **Reasoning**: Thermal infrared data is continuous floating-point temperature data. Saving as `.png` or `.jpg` would truncate this data into 8-bit integers (0-255), destroying fine thermal gradients. Saving as `.tif` is slow to read in PyTorch DataLoaders. `.npy` files load directly into memory with zero precision loss and maximum speed.

## 3. Why Dynamic U-Net?
- **Decision**: Replace the standard 8-block 256x256 U-Net with a `GeneratorUNetDynamic` class.
- **Reasoning**: The original U-Net hardcoded 8 downsampling steps, which assumes a 256x256 input. When the dataset was switched to 128x128 patches (to increase batch size and training speed), the 8-block U-Net crashed because it downsampled the spatial dimensions below 1x1. The dynamic generator calculates `log2(image_size)` to determine exactly how many layers it needs to reach the 1x1 bottleneck safely.

## 4. Why Bilinear Upsampling instead of Transposed Convolution?
- **Decision**: Use `nn.Upsample(mode='bilinear')` followed by `nn.Conv2d` in the Decoder, instead of `nn.ConvTranspose2d`.
- **Reasoning**: Transposed convolutions are notorious for creating "checkerboard artifacts" (grid-like patterns in the generated image). Bilinear upsampling followed by standard convolution entirely avoids this issue while maintaining the same representational power.

## 5. Why Spectral Normalization?
- **Decision**: Use `SpectralNorm` on the Discriminator, and explicitly omit `InstanceNorm`.
- **Reasoning**: Standard GAN training is highly unstable. If the Discriminator learns too fast, its gradients explode, pushing the Generator's gradients to zero (mode collapse). Spectral Normalization forces the Discriminator to be 1-Lipschitz continuous (its gradients cannot exceed a certain slope). Mixing `InstanceNorm` with `SpectralNorm` mathematically invalidates the Lipschitz constraint, which is why it was removed.

## 6. Why React + FastAPI instead of Streamlit?
- **Decision**: Tear out the original Streamlit demo and replace it with a decoupled React frontend and FastAPI backend.
- **Reasoning**: Streamlit is great for quick prototyping but terrible for production web apps. It executes the entire Python script from top to bottom on every user interaction, causing massive UI lag. The React frontend provides instant UI feedback (like the before/after slider), while FastAPI handles the ML inference asynchronously in the background.

## 7. Why Local Percentile Stretching?
- **Decision**: Normalize each input thermal patch using its own 2nd and 98th percentiles, rather than global dataset statistics.
- **Reasoning**: Thermal distributions vary wildly. A patch in the Sahara might range from 30°C to 50°C, while a patch in Antarctica ranges from -40°C to -20°C. If we used global statistics, the Sahara would be pure white and Antarctica pure black, hiding all local details. Local percentile stretching ensures high contrast and detail visibility in *every* patch, regardless of global location.
