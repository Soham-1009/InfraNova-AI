# InfraNova AI — Model Architecture

InfraNova AI uses a **Pix2Pix Conditional GAN**. The model consists of two competing networks: a Generator that tries to create realistic RGB images from thermal inputs, and a Discriminator that tries to tell the difference between real RGB satellite photos and the generated fakes.

## 1. Generator (`GeneratorUNetDynamic`)

The generator is a U-Net encoder-decoder architecture. The "Dynamic" aspect means it automatically calculates the necessary network depth based on the configured `image_size`.

For `image_size = 128`, the generator calculates a depth of 7 (`log2(128) = 7`).

### 1.1 Architecture Table (Depth = 7)

| Layer Name | Type | In Channels | Out Channels | Resolution | Normalization | Activation |
|------------|------|-------------|--------------|------------|---------------|------------|
| `down1` | Conv2d (Stride 2) | 1 (TIR) | 64 | 64×64 | None | LeakyReLU(0.2) |
| `down2` | Conv2d (Stride 2) | 64 | 128 | 32×32 | InstanceNorm2d | LeakyReLU(0.2) |
| `down3` | Conv2d (Stride 2) | 128 | 256 | 16×16 | InstanceNorm2d | LeakyReLU(0.2) |
| `down4` | Conv2d (Stride 2) | 256 | 512 | 8×8 | InstanceNorm2d | LeakyReLU(0.2) |
| `down5` | Conv2d (Stride 2) | 512 | 512 | 4×4 | InstanceNorm2d | LeakyReLU(0.2) |
| `down6` | Conv2d (Stride 2) | 512 | 512 | 2×2 | InstanceNorm2d | LeakyReLU(0.2) |
| `down7` (Bottleneck) | Conv2d (Stride 2) | 512 | 512 | 1×1 | None | LeakyReLU(0.2) |
| `up1` | Upsample+Conv2d | 512 | 512 | 2×2 | InstanceNorm2d | ReLU + Dropout(0.5) |
| `up2` | Upsample+Conv2d | 1024 (512+skip) | 512 | 4×4 | InstanceNorm2d | ReLU + Dropout(0.5) |
| `up3` | Upsample+Conv2d | 1024 (512+skip) | 512 | 8×8 | InstanceNorm2d | ReLU + Dropout(0.5) |
| `up4` | Upsample+Conv2d | 1024 (512+skip) | 256 | 16×16 | InstanceNorm2d | ReLU |
| `up5` | Upsample+Conv2d | 512 (256+skip) | 128 | 32×32 | InstanceNorm2d | ReLU |
| `up6` | Upsample+Conv2d | 256 (128+skip) | 64 | 64×64 | InstanceNorm2d | ReLU |
| `final_up` | Upsample+Conv2d | 128 (64+skip) | 3 (RGB) | 128×128 | None | Tanh |

### 1.2 Key Design Decisions
- **U-Net Skip Connections**: The output of each `down` layer is concatenated with the input of the corresponding `up` layer (e.g., `down6` connects to `up1`). This allows high-frequency spatial details (edges, shapes) to bypass the 1x1 bottleneck.
- **Resize Convolution**: Instead of using `ConvTranspose2d` (which causes checkerboard artifacts), the upsampling blocks use Bilinear `Upsample` followed by a standard `Conv2d` with stride 1.
- **Dropout**: Applied to the first 3 decoder layers. In Pix2Pix, dropout acts as a source of noise to prevent deterministic output, enabling the generator to model a distribution of possible colorizations.

---

## 2. Discriminator (`PatchDiscriminator`)

The discriminator is a **PatchGAN**. Instead of outputting a single "real/fake" probability for the whole 128×128 image, it outputs a grid of scores. Each score evaluates a local 70×70 pixel "patch" of the image. This encourages the generator to produce high-frequency details (textures) that look realistic.

### 2.1 Architecture Table

| Layer Name | Type | In Channels | Out Channels | Resolution | Normalization | Activation |
|------------|------|-------------|--------------|------------|---------------|------------|
| `initial` | Conv2d (Stride 2) | 4 (TIR+RGB) | 64 | 64×64 | SpectralNorm | LeakyReLU(0.2) |
| `block1` | Conv2d (Stride 2) | 64 | 128 | 32×32 | SpectralNorm | LeakyReLU(0.2) |
| `block2` | Conv2d (Stride 2) | 128 | 256 | 16×16 | SpectralNorm | LeakyReLU(0.2) |
| `block3` | Conv2d (Stride 1) | 256 | 512 | 15×15 | SpectralNorm | LeakyReLU(0.2) |
| `final` | Conv2d (Stride 1) | 512 | 1 | 14×14 | SpectralNorm | None |

*Note: For a 256×256 input, the output grid is 30×30. For the current 128×128 input, the output grid is 14×14.*

### 2.2 Key Design Decisions
- **Conditional Input**: The discriminator receives 4 channels: the 1-channel TIR input concatenated with the 3-channel RGB image (either real or fake). This forces the generator to not just output a realistic RGB image, but an RGB image that *matches* the specific thermal input.
- **Spectral Normalization**: SNGAN technique. Constrains the Lipschitz constant of the discriminator, stabilizing adversarial training and preventing the discriminator from overpowering the generator early on. `InstanceNorm` is explicitly omitted because it interferes with Spectral Normalization.
- **Multi-Scale (Optional)**: `MultiScaleDiscriminator` runs two PatchGANs — one on the original image, one on a 2x downsampled version. *(Implemented in code, but currently `multi_scale_disc: false` in config)*.

---

## 3. Forward Pass & Training Flow

1. **Discriminator Update**:
   - Generator produces `fake_rgb` from `ir`.
   - Discriminator scores `cat([ir, real_rgb])` → Loss pushes scores toward 1.0.
   - Discriminator scores `cat([ir, fake_rgb.detach()])` → Loss pushes scores toward 0.0.
   - Discriminator weights are updated (LR = 0.0001).

2. **Generator Update**:
   - Generator produces `fake_rgb` from `ir`.
   - Discriminator scores `cat([ir, fake_rgb])` → GAN Loss pushes scores toward 1.0 (fooling the discriminator).
   - L1 Loss measures absolute pixel difference between `fake_rgb` and `real_rgb`.
   - Perceptual, SSIM, and Chroma losses are calculated between `fake_rgb` and `real_rgb`.
   - Generator weights are updated (LR = 0.0002).
