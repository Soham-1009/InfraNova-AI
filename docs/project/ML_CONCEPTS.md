# InfraNova AI — Machine Learning Concepts

This document explains the core machine learning concepts implemented in the InfraNova AI codebase.

---

## 1. Conditional GAN (cGAN)

**WHAT IS IT?**
A Generative Adversarial Network where both the Generator (creating images) and the Discriminator (judging images) are conditioned on some extra information. In this case, the condition is the input thermal image.

**WHY DOES THIS PROJECT USE IT?**
If we just used a standard GAN, the model would generate random, realistic-looking satellite photos. We don't want random photos; we want the specific RGB photo that corresponds to the input thermal map. By passing the thermal map to both networks, the GAN learns to translate the specific input into its specific output.

**WHERE IS IT IMPLEMENTED?**
`src/models/pix2pix/pix2pix.py`
The `discriminate` function concatenates the IR input and the RGB image before passing it to the discriminator: `x = torch.cat([ir, rgb], dim=1)`.

---

## 2. PatchGAN

**WHAT IS IT?**
A specific discriminator architecture that classifies N×N patches of an image as real or fake, rather than classifying the entire image with a single score.

**WHY DOES THIS PROJECT USE IT?**
L1 loss is great at capturing low-frequency information (general shapes, colors) but produces blurry images. PatchGAN forces the generator to focus on high-frequency details (textures, edges, roads, buildings) at a local scale, resulting in sharper images.

**HOW DOES IT WORK?**
It's a Fully Convolutional Network without dense layers. For a 256x256 input, a standard PatchGAN outputs a 30x30 grid of probabilities. Each probability corresponds to a 70x70 pixel receptive field in the original image.

**WHERE IS IT IMPLEMENTED?**
`src/models/pix2pix/discriminator.py:PatchDiscriminator`

---

## 3. Dynamic U-Net

**WHAT IS IT?**
An encoder-decoder architecture with skip connections between mirrored layers. "Dynamic" means it recursively calculates how many layers it needs based on the input image size.

**HOW DOES IT WORK?**
The encoder repeatedly cuts the spatial resolution in half while doubling the channel count, creating a 1x1 "bottleneck" of dense features. The decoder reverses this. The skip connections concatenate the output of an encoder layer directly to the input of the corresponding decoder layer, allowing spatial details to bypass the bottleneck.

**WHERE IS IT IMPLEMENTED?**
`src/models/pix2pix/generator_dynamic.py`

**WHAT HAPPENS IF WE REMOVE SKIP CONNECTIONS?**
The network becomes a standard autoencoder. Because the bottleneck is 1x1, all spatial information (where exactly a road or river is) is lost. The output would be a blurry, blobby mess of colors.

---

## 4. VGG Perceptual Loss

**WHAT IS IT?**
Instead of comparing pixels directly (like L1 loss), perceptual loss passes both the generated image and the real image through a pretrained image classifier (VGG19) and compares their internal feature maps.

**WHY DOES THIS PROJECT USE IT?**
If the generator produces a road that is shifted by just one pixel from the real image, pixel-wise L1 loss penalizes it heavily. Perceptual loss understands that both images contain a "road" and penalizes the network much less. This encourages the model to focus on generating realistic features rather than exact pixel replication.

**WHERE IS IT IMPLEMENTED?**
`src/training/losses.py:VGGPerceptualLoss`

---

## 5. Structural Similarity Index Measure (SSIM) Loss

**WHAT IS IT?**
A loss function based on the human visual system's perception of structural information, luminance, and contrast.

**WHY DOES THIS PROJECT USE IT?**
While GANs make images look realistic and L1 makes them mathematically close, SSIM ensures that the structure (edges, shapes of buildings/fields) aligns well with the ground truth. It acts as a bridge between the blurriness of L1 and the sometimes "hallucinated" textures of a GAN.

**WHERE IS IT IMPLEMENTED?**
`src/training/losses.py:SSIMLoss`

---

## 6. Spectral Normalization (SNGAN)

**WHAT IS IT?**
A technique applied to the weights of the discriminator's convolutional layers that divides the weight matrix by its largest singular value.

**WHY DOES THIS PROJECT USE IT?**
It enforces a mathematical property called the Lipschitz constraint. Practically, it prevents the discriminator's gradients from exploding. In GANs, if the discriminator gets too good too fast, it just outputs "0" for all fakes, giving the generator no useful gradient signal to learn from. Spectral Normalization keeps the discriminator's judgments smooth and stable.

**WHERE IS IT IMPLEMENTED?**
`src/models/pix2pix/discriminator.py:DiscBlock` (using `torch.nn.utils.spectral_norm`)

---

## 7. Test-Time Augmentation (TTA)

**WHAT IS IT?**
An inference technique where the input image is passed through the model multiple times with different geometric transformations (e.g., flipped, rotated). The model's predictions are then inversely transformed and averaged together.

**WHY DOES THIS PROJECT USE IT?**
CNNs are not perfectly rotation or flip invariant. By generating predictions for the image at 4 different orientations and averaging them, artifacts are smoothed out and the overall colorization quality is noticeably improved.

**WHERE IS IT IMPLEMENTED?**
`demo/inference.py:predict()` (lines 162-188)

---

## Documented but NOT Implemented Concepts

- **Object Detection / YOLO**: Mentioned in early project goals, but no implementation exists in the codebase (the `src/detection/` directory is empty).
- **Super-Resolution**: The dataset generates 64x64 (`tir_200m.npy`) patches intended for super-resolution, but the model and training pipeline only implement colorization (`tir_100m` -> `rgb_100m`).
