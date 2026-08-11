# InfraNova AI — Loss Functions

This document explains the objective functions used to train the Pix2Pix Generator. All losses are calculated in `src/training/losses.py` and combined via `CombinedLoss`.

## 1. Objective Composition

The final Generator loss is a weighted sum of five individual components. The weights (lambdas) are defined in `configs/config.yaml`.

```text
Total Loss = (λ_adv × GANLoss) 
           + (λ_l1 × L1Loss) 
           + (λ_perc × PerceptualLoss) 
           + (λ_ssim × SSIMLoss) 
           + (λ_chroma × ChromaLoss)
```

**Current Configuration:**
- `lambda_adv`: 1.0
- `lambda_l1`: 10.0
- `lambda_perc`: 10.0
- `lambda_ssim`: 5.0
- `lambda_chroma`: 2.0
- *`lambda_feat`: 0.0 (Disabled)*

## 2. Component Details

### 2.1 Adversarial GAN Loss (`GANLoss`)
- **Purpose**: Forces the generator to produce images that are indistinguishable from real satellite imagery.
- **Mode**: Binary Cross Entropy (`bce`). Evaluates the PatchGAN discriminator's output against a target matrix of 1.0s.
- **Why it matters**: Without this, the model produces blurry, safe averages. GAN loss creates the high-frequency textures (trees, buildings) that make it look real.

### 2.2 Pixel L1 Loss (`PixelL1Loss`)
- **Purpose**: Enforces low-frequency correctness (overall color and shape).
- **Calculation**: Mean Absolute Error between generated pixels and real pixels.
- **Why it matters**: Ensures the generated image actually matches the geography of the thermal input, rather than just generating a random realistic landscape.

### 2.3 VGG Perceptual Loss (`VGGPerceptualLoss`)
- **Purpose**: Measures semantic and stylistic difference.
- **Calculation**: Passes both real and generated images through a pre-trained VGG19 network. Computes L1 distance between the feature maps at layers `relu1_2`, `relu2_2`, `relu3_4`, and `relu4_4`.
- **Why it matters**: A road shifted by 1 pixel has a huge L1 penalty but a tiny Perceptual penalty. It prevents the model from being overly penalized for slight structural shifts that still look completely realistic.

### 2.4 Structural Similarity Loss (`SSIMLoss`)
- **Purpose**: Preserves structural integrity (edges, boundaries).
- **Calculation**: Computes `(1 - SSIM(fake, real)) / 2`. Uses a differentiable 11x11 Gaussian filter.
- **Why it matters**: Satellite imagery relies heavily on structural shapes (agricultural fields, coastlines, city blocks). SSIM explicitly rewards matching these structures.

### 2.5 Chroma Loss (`ChromaLoss`)
- **Purpose**: Prevents the model from generating desaturated, brownish images (a common issue in conditional GANs mapping 1-channel to 3-channels).
- **Calculation**: Converts RGB tensors to YCbCr approximation, isolates the Cb and Cr (color) channels, and computes the L1 difference between real and fake chroma.
- **Why it matters**: Forces the generator to aggressively learn the color distribution rather than playing it safe with gray tones.

## 3. Unused Losses

### Feature Matching Loss (`FeatureMatchingLoss`)
- **Calculation**: Compares the internal feature maps of the *Discriminator* when evaluating real vs fake images.
- **Status**: Implemented but disabled (`lambda_feat: 0.0`). In early testing, it did not provide significant benefits over VGG Perceptual Loss and slowed down training.
