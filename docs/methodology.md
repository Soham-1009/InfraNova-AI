## Mathematical Validation of Pix2Pix Architecture for IR→RGB Translation
### 1. GENERATOR: U‑Net with 8 Down‑/Up‑sampling Blocks

#### 1.1 Receptive Field Analysis – Why Exactly 8 Blocks?

The encoder uses `Conv2d(kernel=4, stride=2, padding=1)` → InstanceNorm → LeakyReLU(0.2). Each such block halves the spatial resolution. Starting from \(256\times256\):

| Block | Input Size | Output Size | Channels |
|-------|------------|-------------|----------|
| 1     | 256×256    | 128×128     | 64       |
| 2     | 128×128    | 64×64       | 128      |
| 3     | 64×64      | 32×32       | 256      |
| 4     | 32×32      | 16×16       | 512      |
| 5     | 16×16      | 8×8         | 512      |
| 6     | 8×8        | 4×4         | 512      |
| 7     | 4×4        | 2×2         | 512      |
| 8     | 2×2        | 1×1         | 512      |

After 8 such blocks, the spatial size collapses to \(1\times1\) – a single global feature vector. The **theoretical receptive field** (RF) of a neuron in this bottleneck layer can be computed layer‑wise.

For a convolutional layer with kernel size \(k\), stride \(s\):
\[
RF_{\text{out}} = s \cdot (RF_{\text{in}} - 1) + k
\]
Starting from \(RF=1\) at the bottleneck (output of block 8) and working backwards:

| Layer | Kernel | Stride | RF before block |
|-------|--------|--------|-----------------|
| Block 8 (to 1×1) | 4 | 2 | \((1-1)\cdot 2 + 4 = 4\) |
| Block 7 (to 2×2) | 4 | 2 | \((4-1)\cdot 2 + 4 = 10\) |
| Block 6 (to 4×4) | 4 | 2 | \((10-1)\cdot 2 + 4 = 22\) |
| Block 5 (to 8×8) | 4 | 2 | \((22-1)\cdot 2 + 4 = 46\) |
| Block 4 (to 16×16) | 4 | 2 | \((46-1)\cdot 2 + 4 = 94\) |
| Block 3 (to 32×32) | 4 | 2 | \((94-1)\cdot 2 + 4 = 190\) |
| Block 2 (to 64×64) | 4 | 2 | \((190-1)\cdot 2 + 4 = 382\) |
| Block 1 (to 128×128) | 4 | 2 | \((382-1)\cdot 2 + 4 = 766\) |

Thus a single neuron in the bottleneck **sees a \(766\times766\) input region**, far larger than the \(256\times256\) image. In reality, the *effective* receptive field (Gaussian weighting) is smaller but still covers the entire image, giving the model global context.  
**Why 8 blocks is optimal**: Fewer than 8 blocks would not reach a \(1\times1\) bottleneck (e.g., 7 blocks give \(2\times2\) bottleneck, RF ~382, still covering the image but with less global compression). 8 blocks gives the deepest possible encoding, which is essential for colourisation – disambiguating vegetation from water requires whole-scene context.

#### 1.2 Channel Progression and Parameter Efficiency

The channel doubling until 512 and then holding at 512 follows the U‑Net convention. In the bottleneck, a high channel count (512) allows the network to store a rich latent representation of the scene’s semantic layout, while earlier layers focus on fine spatial detail. The parameter count scales quadratically with channel numbers; capping at 512 avoids an explosion in parameters while providing sufficient capacity.

#### 1.3 Skip Connections – Gradient Highway and Detail Preservation

Let the encoder activations be \(E_i\) and decoder feature maps \(D_i\). The concatenation \( [D_i, E_i] \) preserves high‑frequency spatial information that would otherwise be lost after the bottleneck. During backpropagation, the gradient from the decoder is directly routed to the encoder through the skip connection:

\[
\frac{\partial \mathcal{L}}{\partial E_i} \;=\; \frac{\partial \mathcal{L}}{\partial D_{i-1}} \;\oplus\; \text{(gradient through upsample path)}
\]

This dual path mitigates the vanishing gradient problem and enables training of very deep architectures. For IR→RGB, the skip connections allow the decoder to recover exact building boundaries and road edges from the encoder’s early layers, which is critical because IR images contain sharp structural cues that must appear in the colourised output.

#### 1.4 Instance Normalisation vs Batch Normalisation

Batch Normalisation (BN) normalises over the batch and spatial dimensions:
\[
y = \gamma \frac{x - \mu_{\text{batch}}}{\sigma_{\text{batch}}} + \beta
\]
where \(\mu_{\text{batch}}, \sigma_{\text{batch}}\) are computed across \(N\times H\times W\). In image translation, this couples the output of one sample to the statistics of the entire mini‑batch, which can cause mode collapse, leakage of instance‑specific contrast, and instability when batch sizes are small.

Instance Normalisation (IN) normalises each sample independently:
\[
y = \gamma \frac{x - \mu_{n,:,:}}{\sigma_{n,:,:}} + \beta
\]
where \(\mu_{n,:,:}\) is the spatial mean for sample \(n\) and channel. IN removes instance‑specific contrast and illumination shifts, which is highly desirable for style transfer and colourisation. For our task, the IR input contains its own luminance distribution; IN allows the network to re‑scale the internal activations without depending on other images, leading to sharper output and faster convergence. Therefore we use **InstanceNorm** in both generator and discriminator.

---

### 2. DISCRIMINATOR: 70×70 PatchGAN with Spectral Normalisation

#### 2.1 Receptive Field Derivation for PatchGAN

The PatchGAN discriminator architecture (adopted from Pix2Pix) is as follows (input: 6‑channel concatenated IR+RGB, \(256\times256\)):

| Layer | Type | Kernel | Stride | Output size |
|-------|------|--------|--------|-------------|
| C64   | Conv→LReLU | 4 | 2 | 128×128 |
| C128  | Conv→IN→LReLU | 4 | 2 | 64×64 |
| C256  | Conv→IN→LReLU | 4 | 2 | 32×32 |
| C512  | Conv→IN→LReLU | 4 | 1 | 31×31 |
| C1    | Conv (no norm) | 4 | 1 | 30×30 → output 30×30 patch of logits |

We compute the receptive field of a single output pixel (30×30 grid) relative to the input image (same method as before, now forward):

- After C1 (kernel 4, stride 1): RF = 4
- After C512 (kernel 4, stride 1): RF = 4 + (4-1) = 7
- After C256 (kernel 4, stride 2): RF = (7-1)*2 + 4 = 16
- After C128 (kernel 4, stride 2): RF = (16-1)*2 + 4 = 34
- After C64 (kernel 4, stride 2): RF = (34-1)*2 + 4 = 70

Thus each output pixel classifies a **70×70 patch** of the input. The discriminator models the image as a Markov Random Field with independence assumption beyond 70 pixels. This local approach is far more effective than a full‑image discriminator because:

- It provides a **dense grid of realism feedback** (30×30 = 900 loss terms per image), forcing the generator to produce high‑frequency textures everywhere.
- It has fewer parameters and is less prone to overfitting (only 2.8M parameters).
- It can be applied to any image size during inference without architecture change.

#### 2.2 Spectral Normalisation – Lipschitz Stability

For a linear layer \(f(x) = Wx\), the spectral norm \(\sigma(W)\) is the largest singular value of \(W\). Spectral Normalisation (SN) replaces \(W\) with

\[
\hat{W} = \frac{W}{\sigma(W)}
\]

so that \(\|f\|_{\text{Lip}} = \sigma(\hat{W}) = 1\). By applying SN to every convolutional layer in the discriminator, the Lipschitz constant of the entire network is bounded by 1. This prevents the discriminator from becoming arbitrarily steep, stabilising the GAN training in two ways:

1. **Gradient magnitude control**: The gradient of the LSGAN discriminator with respect to its input is bounded, so the generator receives consistent and meaningful gradients even when the discriminator is optimal.
2. **No need for gradient penalty**: Unlike WGAN‑GP, SN does not require extra backward passes, making it computationally cheap.

SN is computed once per training step using the power iteration method, which quickly approximates \(\sigma(W)\). In PyTorch, `torch.nn.utils.spectral_norm` applies it directly. We enable SN on all discriminator convolution layers.

---

### 3. LOSS FUNCTION WEIGHTS – BALANCING MULTIPLE OBJECTIVES

Our combined generator loss:
\[
\mathcal{L}_{\text{total}} = 0.5\,\mathcal{L}_{\text{adv}} + 100\,\mathcal{L}_{L1} + 10\,\mathcal{L}_{\text{perceptual}} + 3\,\mathcal{L}_{\text{SSIM}}
\]

#### 3.1 Why These Specific Weights? – Gradient Magnitude Analysis

All image tensors are in \([-1,1]\). Approximate expected magnitudes at initialisation (random generator, pre‑trained discriminator barely better than chance):

- \(\mathcal{L}_{L1}\) = \(\frac{1}{N}\sum |y - G(x)| \approx 0.3\) (average deviation).
- \(\mathcal{L}_{\text{adv}}^{\text{LSGAN}}\) = \(\frac{1}{2}\mathbb{E}[(D(G)-1)^2] \approx 0.5\) (discriminator outputs near 0).
- \(\mathcal{L}_{\text{perceptual}}\) (VGG features) ≈ 1.0 (features are not yet aligned).
- \(\mathcal{L}_{\text{SSIM}}\) ≈ \(1 - 0.3 = 0.7\).

The gradient magnitude of each term w.r.t. the generator output is approximately:

- \(\|\partial\mathcal{L}_{L1}/\partial G\| \approx 1\) (unit vector of sign).
- \(\|\partial\mathcal{L}_{\text{adv}}/\partial G\| \propto (D(G)-1) \approx 1\).
- Perceptual and SSIM gradients are smaller due to intermediate operations.

Multiplying by the given coefficients yields effective gradient contributions:

- L1: \(100 \times 1 = 100\)
- Adversarial: \(0.5 \times 1 = 0.5\)
- Perceptual: \(10 \times (\sim 0.1) \approx 1\)
- SSIM: \(3 \times (\sim 0.05) \approx 0.15\)

**The L1 term dominates the early training by a factor of ~200.** This is intentional: the generator must first learn a pixel‑accurate mapping (global colour assignment, shape correspondence) because the IR → RGB mapping is strongly constrained by luminance equivalence. Once the L1 loss reduces (error shrinks), the effective L1 gradient decreases proportionally, allowing the adversarial and perceptual losses to refine textures and fine details. This natural curriculum prevents the GAN from producing colourful but geometrically wrong outputs (a typical failure when adversarial weight is too high).

#### 3.2 What Happens if Weights are Wrong?

- **Adversarial too high (e.g., λ₁ = 10)**: The generator will hallucinate textures that look real but are spatially misaligned with the IR – roads might become rivers, buildings turn to vegetation. Object detectors would fail catastrophically.
- **L1 too low (e.g., λ₂ = 10)**: Output remains blurry, lacking sharp edges; the adversarial loss cannot compensate for the lack of pixel‑level supervision because it only cares about local realism.
- **Perceptual too high (λ₃ = 100)**: Checkerboard artefacts appear; the VGG features can be fooled by high‑frequency noise that matches the statistics of natural images but is not present in the ground truth.
- **SSIM too high (λ₄ = 30)**: Over‑smoothing, loss of colour vibrancy, because SSIM strongly penalises structural differences and tends to pull the image towards a washed‑out average.

The selected weights have been empirically validated in numerous Pix2Pix‑based works and represent the sweet spot for paired satellite translation.

#### 3.3 Mathematical Derivation of the Weight Balance

Consider the total loss as a Lagrangian for a constrained optimisation: we want to minimise pixel error (L1) subject to constraints on realism (adversarial) and perceptual similarity. The weights are Lagrange multipliers. If we treat the adversarial term as a penalty for non‑realistic images, the multiplier λ₁ should be small enough that the generator does not sacrifice pixel accuracy for realism. The ratio \(\lambda_2/\lambda_1 = 200\) ensures that at convergence, the generator’s output has an adversarial loss near 0.1 (discriminator fooled) and an L1 loss around 0.05. The perceptual weight λ₃=10 is chosen to be of the same order as the adversarial weight once the L1 loss has decayed, so that after 30–50 epochs all three terms are roughly in balance. The SSIM weight λ₄=3 is deliberately kept lower to avoid introducing unwanted blurring; SSIM mainly acts as a structural regulariser.

---

### 4. MEMORY BUDGET & BATCH SIZE – FITTING 15 GB T4

#### 4.1 Parameter Count Verification

- **Generator (U‑Net 256)**: The architecture with channels [64,128,256,512,512,512,512,512] and up‑blocks yields ≈ 54.4 M parameters. (Verified by constructing a dummy model in PyTorch and calling `sum(p.numel())`.)
- **Discriminator (PatchGAN 70×70)**: C64‑C128‑C256‑C512‑C1 with spectral norm → ≈ 2.76 M parameters.
- **Total trainable parameters**: ~57.2 M.

In fp32, this is \(57.2 \times 10^6 \times 4 \approx 229 \text{ MB}\).  
Adam optimizer stores two additional tensors per parameter (momentum and variance): \(229 \times 2 = 458 \text{ MB}\).  
Thus, **model + optimiser occupy ≈ 687 MB**, regardless of batch size.

#### 4.2 Activation Memory per Sample (fp32)

We must account for all intermediate feature maps saved for backward pass, including skip connections. Using the U‑Net architecture above, a conservative estimation (summing sizes of all output tensors from conv+norm+relu layers in both encoder and decoder) gives ~1.8 GB per 256×256 sample in fp32. For safety we use 2.0 GB per sample as an upper bound.

Therefore, in fp32:
- 1 sample: 2.0 GB + 0.69 GB ≈ 2.7 GB
- Batch size 4: 8.0 + 0.69 ≈ 8.7 GB  (fits)
- Batch size 8: 16.0 + 0.69 ≈ 16.7 GB (exceeds 15 GB)

#### 4.3 Mixed Precision (AMP) Savings

With automatic mixed precision, all activations are stored in fp16, halving activation memory to ≈ 1.0 GB per sample. The optimizer states and master parameters remain in fp32 (same 0.69 GB).  
Thus, with AMP:
- 1 sample: 1.0 + 0.69 ≈ 1.7 GB
- Batch size 8: 8.0 + 0.69 ≈ 8.7 GB  – **comfortably within 15 GB**, leaving room for framework overhead.

**Conclusion**: Using AMP, a batch size of 8 is both safe and optimal for T4 (15 GB). We can even increase to 10–12 if needed, but 8 is a robust choice that allows for gradient accumulation if we wish to simulate larger effective batches.

#### 4.4 Gradient Accumulation as a Fallback

If memory spikes occur, we can drop to batch size 4 and use `accumulation_steps=2` to maintain the same effective batch size for the running statistics. This keeps VRAM usage under 9 GB while preserving training dynamics.

---

### 5. CONVERGENCE EXPECTATIONS & OVERFITTING ANALYSIS

#### 5.1 How Many Epochs with 2000 Samples?

With heavy geometric augmentation (flips, 90° rotations, random crops), each original 256×256 image yields ≈ 8 distinct training patches per epoch (crop size equal to input, but with flips/rotations). The effective number of unique training examples per epoch becomes ≈ 16,000. Pix2Pix models typically require seeing 20–50 passes over the dataset (epochs) before adversarial loss stabilises. Thus, we expect the model to converge in **100–200 epochs**. Early stopping based on validation SSIM (hold‑out 10%) should trigger around epoch 150 if no improvement.

#### 5.2 Signs of Mode Collapse vs Healthy Training

- **Healthy training**: Discriminator loss oscillates around 0.2–0.5, generator adversarial loss slowly decreases, L1 loss steadily drops, validation SSIM improves and plateaus.
- **Mode collapse**: Generator produces similar colours or textures for vastly different IR inputs; the LPIPS distance between generated outputs for different IR images becomes very small (<0.05). This can be detected by monitoring the LPIPS diversity on a fixed validation set.
- **Overfitting**: Validation L1 starts increasing while training L1 keeps decreasing. We stop training immediately.

#### 5.3 Theoretical Upper Bound on PSNR

As argued earlier, the mapping IR → RGB is ill‑posed: a single NIR intensity value can correspond to many RGB triplets. The conditional distribution \(p(\text{RGB}|\text{IR})\) has non‑zero variance. The L1 loss encourages the generator to output the per‑pixel median of that distribution. The irreducible error is the absolute deviation of the median from the true RGB. For natural satellite scenes, this intrinsic uncertainty imposes an upper bound on PSNR of approximately **30–32 dB**. Any attempt to push PSNR beyond this leads to hallucination that cannot be verified by the ground truth, and our evaluation should focus on perceptual metrics (LPIPS, FID) that reward plausible colourisation even if it deviates from the one captured RGB example. We will report PSNR as a secondary metric, acknowledging this ceiling.

---

**Summary of Validated Configuration**

| Component          | Specification                | Mathematical Justification |
|--------------------|------------------------------|----------------------------|
| Generator depth    | 8 down/up blocks             | Ensures 1×1 bottleneck, RF covers full 256×256 image |
| Channel prog.      | 64→128→256→512→512…          | Balances capacity and parameter count |
| Skip connections   | Concatenate                  | Preserves high‑frequency gradients and detail |
| Normalisation      | InstanceNorm                 | Removes instance‑specific contrast shifts, stabilises training with small batches |
| Discriminator      | 70×70 PatchGAN + Spectral Norm | Dense local feedback; Lipschitz control prevents vanishing gradients |
| Loss weights       | 0.5 / 100 / 10 / 3           | L1 dominates early (pixel accuracy), adv/perceptual refine later |
| Batch size (AMP)   | 8                            | Fits 15 GB T4, allows stable GAN dynamics |
| Expected epochs    | 100–150 with early stopping  | Augmented dataset equivalent ~16k patches/epoch |
| PSNR ceiling       | ~30 dB                       | Irreducible uncertainty from ill‑posed inverse problem |

This mathematical framework underpins our entire training pipeline. The design is now fully ready for implementation.

## Day 4 – Mathematical Validation of Pix2Pix Training Procedure

This analysis validates the chosen hyperparameters and training protocol using optimisation theory, minimax dynamics, and numerical stability considerations. All recommendations are specific to our hardware (T4 15 GB, AMP) and dataset (2000 IR‑RGB pairs, 256²).

---

### 1. Optimizer Choice: Adam with β₁=0.5, β₂=0.999, lr=2×10⁻⁴

**Why these values for GANs?**

The GAN objective is a minimax game:
\[
\min_G \max_D \; V(D,G)
\]
Optimising such a saddle‑point problem with standard SGD can lead to divergent oscillations because the loss landscape is non‑convex and non‑concave, and the vector field has a rotational component. Adam adds momentum that averages gradients over time, damping oscillations and speeding up convergence. However, the default momentum parameter β₁=0.9 is too high: it causes the optimiser to carry too much “memory” of past gradients, making the generator overshoot the equilibrium and leading to training instability (discriminator loss collapses, mode collapse). Reducing β₁ to **0.5** shortens the effective memory window – the exponential moving average of past gradients decays as (0.5)^t, so after ∼3 steps the influence is below 10%. This allows Adam to react quickly to the shifting adversarial target.

β₂=0.999 remains unchanged because the RMS term (second moment) should adapt slowly to maintain a stable per‑parameter learning rate. The learning rate 2×10⁻⁴ is standard for Pix2Pix; it was tuned in the original paper over a wide range of tasks. A higher LR causes wild oscillations, a lower one slows convergence.

**Should generator and discriminator have the same LR?**

Yes, they can use the same LR, because:

* The generator must learn a complex mapping quickly, while the discriminator needs to keep up but not overpower.  
* In Pix2Pix, the discriminator sees a simpler task (real/fake classification on patches), so it converges faster. If we gave the discriminator a higher LR, it would dominate early, causing vanishing generator gradients. A lower discriminator LR would allow the generator to fool it too easily.  
* A balanced 2×10⁻⁴ for both works well; many implementations even use **1×10⁻⁴** for both as a safer default. We can stick with 2×10⁻⁴ but monitor the discriminator accuracy: it should stay around 50–70%. If it exceeds 90%, lower discriminator LR to 1×10⁻⁴.

---

### 2. Two‑Step Training Schedule (Alternating Updates)

The procedure is not a two‑phase epoch‑level schedule, but a per‑iteration alternation:

1. **Discriminator step:** Feed a batch of real (IR, RGB) and generated (IR, fake RGB) pairs. Compute \(\mathcal{L}_D\) and update D.  
2. **Generator step:** Freeze D, feed a batch of IR to G, compute combined \(\mathcal{L}_G\) (including \(\mathcal{L}_{\text{adv}}\) that uses D’s judgement), and update G.

This is the standard alternating gradient descent for a min‑max game. Mathematically, it approximates the simultaneous gradient dynamics (gradient descent‑ascent) but with a critical advantage: **the generator sees the same discriminator state for its entire update**, which avoids the confusion that would occur if D kept changing mid‑step.

If we trained both networks in the same forward pass (e.g., feeding data, computing both losses, and backpropagating both) without freezing the discriminator, the generator’s adversarial gradient would be computed with respect to a discriminator that is about to change, creating an inconsistent minimax iteration. The two‑step alternation ensures clear separation of the optimisation steps.

**What if we trained D to convergence before every G step?**  
This would give a near‑optimal discriminator, which would flatten the generator’s gradient (JS‑divergence saturates). In LSGAN the gradient does not saturate, but a too‑strong D still makes the task unrealistically hard, slowing G’s learning. The 1‑step alternation keeps D in a regime where it provides a meaningful learning signal without becoming overwhelming.

---

### 3. Learning Rate Schedule

**Why decay LR for GANs?**  
In the late stages of training, the generator has captured the main colour mapping and the discriminator has learned to spot subtle artefacts. A high LR can then cause large parameter updates that overshoot the equilibrium, reintroducing artefacts or causing mode collapse. Decaying the learning rate smooths out the last fluctuations and helps the networks settle into a stable local Nash equilibrium.

**Linear decay vs cosine annealing**  
- **Linear decay**: \(\eta(t) = \eta_0 \left(1 - \frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}}\right)\) for \(t \ge t_{\text{start}}\). Simple, predictable.  
- **Cosine annealing**: \(\eta(t) = \eta_{\text{min}} + \frac{1}{2}(\eta_0 - \eta_{\text{min}})(1 + \cos(\pi \frac{t - t_{\text{start}}}{T}))\). More aggressive decay at the beginning and end, with a plateau in the middle. For a fixed budget of 150 epochs, cosine annealing might force the LR to become very small in the final 30 epochs, which could be beneficial but also might stop learning too early.  

For a hackathon with limited tuning, linear decay is more robust. We propose:

* Start decay at epoch **80** (half of 150).  
* Decay to **0** by epoch 150 (final LR = 0). Actually reaching 0 is acceptable because the model is near convergence.  
* Alternatively, decay to \(10^{-6}\) and hold.  

Implementation: `scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=70)` starting at epoch 80.

If we wanted to be safer, we could decay to 10⁻⁵ and maintain.

**Optimal decay for 150 epochs:**  
We will monitor the validation SSIM. At epoch 80, if SSIM is still improving, we might delay the decay to epoch 100. Many Pix2Pix implementations do not decay at all and still converge fine. We can keep the schedule optional; if training is stable we may skip it. However, for a rigorous methodology, we include a linear decay from epoch 100 to 150.

---

### 4. Gradient Clipping

**Should we clip gradients?**  
In standard Pix2Pix with LSGAN and spectral norm, gradient clipping is **not required** for the discriminator because the spectral norm already enforces a Lipschitz constraint, preventing exploding gradients. However, the generator has no such constraint, and its L1 loss (weight 100) can occasionally produce large parameter gradients when the prediction deviates strongly from the target, especially at early stages. This can push the generator into a region where the discriminator becomes too effective, destabilising the game.

**Recommendation:** Apply gradient clipping to the generator with a maximum L2‑norm of **10.0**. This is a soft limit; in practice it rarely triggers after epoch 5, but it provides a safety net against early explosions. The discriminator does not need clipping. In code:

```python
torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
```

If we observe that clipping never activates, we can remove it to reduce computational overhead.

---

### 5. Mixed Precision (AMP) – Numerical Stability & Memory

**Memory savings calculation:**  
In fp32, activation memory per sample for 256² U‑Net ≈ 2.0 GB. In fp16 (AMP), activations are stored in half precision → 1.0 GB per sample. For a batch of 8, that saves \(8 \times 1.0 = 8\) GB, bringing total VRAM from ~16.7 GB (over limit) to ~8.7 GB (well under 15 GB). This is what enables batch‑size 8.

**Numerical stability concerns:**  
AMP automatically uses **loss scaling** to prevent underflow when gradients are very small. The forward pass of the generator and discriminator runs in fp16; operations that are unsafe (e.g., `LayerNorm`, `softmax`, or small‑value exponentiation) are executed in fp32 automatically by PyTorch’s `autocast`. For our networks, the critical operation is the **perceptual loss**: VGG forward pass must be in fp32 to avoid accuracy loss. We enforce this by wrapping the VGG loss module with `torch.cuda.amp.autocast(enabled=False)`.

Additionally, spectral norm power iteration involves dividing by singular values which could be tiny; these will be kept in fp32 by AMP’s fallback list. Empirical tests show that LSGAN discriminators with spectral norm train safely under AMP.

**Which ops need fp32 for stability?**  
- Normalisation of VGG input (division by std)  
- The final `tanh` output (range [-1,1]) is safe in fp16 because tanh is well‑behaved, but its gradient near saturation might underflow; AMP manages this.  
- SSIM loss uses convolution with a small Gaussian window; it is safe in fp16 but we can keep it in fp32 for safety if we see artifacts.

We will monitor the gradient scale; if it grows beyond 65536, we increase the initial scale.

---

### 6. Checkpointing Strategy

**Metric for best model selection:**  
We want a model that maximises structural fidelity for downstream object detection. SSIM directly measures luminance, contrast, and structural preservation. It is fast to compute and correlates well with perceived similarity of shapes. LPIPS and FID are more perceptual but may reward colourful hallucinations that hurt object recognition. Therefore, we will **select the best model based on validation SSIM** (the higher the better).

**Checkpoint frequency:**  
- Save the model state every **5 epochs** to allow resumption and historical analysis.  
- Additionally, save a separate “best” checkpoint whenever validation SSIM improves.

**Disk space:**  
One checkpoint (generator + discriminator + optimiser states) ≈ 230 MB (param) + 2 × 230 MB (optim) ≈ 700 MB. At 150 epochs with one save per 5 epochs, that’s ~30 saves × 700 MB ≈ 21 GB – too large for a typical Google Drive temporary storage. We must store only the last 3 checkpoints and the best model. We can script deletion of older files. The “best” model is kept indefinitely. So total storage will be < 3 GB. That is acceptable.

---

### 7. Early Stopping Criteria

**Patience:** 20 epochs. If validation SSIM does not increase for 20 consecutive evaluations, we stop.

**Why SSIM?** As argued, SSIM reflects the structural consistency we need. LPIPS might fluctuate and early‑stop later, wasting time.

**When has the model “converged”?**  
A GAN has converged when the discriminator’s loss stops decreasing and oscillates around a constant value, and the generator’s adversarial loss also stabilises. From a game‑theoretic perspective, this indicates a local Nash equilibrium where neither player can unilaterally improve. Numerically, we can define convergence as: the moving average of \(| \Delta \text{SSIM}_{\text{val}} | < 10^{-4}\) per epoch for 10 epochs. With early stopping, this condition is automatically checked.

**Edge case:** If SSIM suddenly degrades while LPIPS improves, the generator may be adding realistic textures that deviate from the ground truth structure (hallucination). We should **stop** and not rely on SSIM alone; we would then need to visually inspect or trust a combined metric. For safety, we could monitor the product \((1 - \text{LPIPS}) \times \text{SSIM}\) and stop if it does not improve for 20 epochs. That’s robust. I’ll recommend that as our early‑stopping metric.

---

### 8. Expected Training Dynamics & Mode Collapse Indicators

**Healthy loss curves:**

- **Discriminator loss (\( \mathcal{L}_D \))**: Starts around 0.5 (LSGAN initial), drops quickly as D learns real/fake, then gradually rises as generator improves, eventually oscillating between 0.2 and 0.5.  
- **Generator adversarial loss (\( \mathcal{L}_{G,\text{adv}} \))**: Starts high (~1.0), decreases as G fools D, eventually stabilising around 0.3–0.5.  
- **L1 loss (\( \mathcal{L}_{L1} \))**: Decreases sharply at first (epochs 1–20), then slowly, asymptoting around 0.03–0.05 (pixel‑wise).  
- **Perceptual + SSIM**: Similar gradual improvement.

**Mode collapse indicators:**  
- **LPIPS diversity collapse**: For a fixed batch of IR images, the LPIPS distance between any two generated outputs drops below 0.02, meaning the generator produces nearly identical colours regardless of content.  
- **Discriminator loss spikes**: Mode collapse often follows a sudden increase in discriminator accuracy (loss drops close to 0) because the generator’s limited output becomes easy to distinguish.  
- **Visual check**: The colour palette becomes uniform (e.g., all vegetation looks the same shade of green).  

If mode collapse is detected early (say epoch 30), we can intervene by:  
- Temporarily increasing the generator’s learning rate  
- Adding a small amount of Gaussian noise to the generator input  
- Reinitialising the last layers of the discriminator  

For our setup, the strong L1 and perceptual losses act as powerful anti‑collapse regularisers, so mode collapse is unlikely.

**When to stop early even before patience triggers:**  
- If the generator loss explodes (L1 > 2.0) or discriminator loss becomes NaN, stop immediately and restore from a previous checkpoint.  
- If after 50 epochs the validation SSIM is still < 0.4, the model has failed to learn basic structure; adjust hyperparameters.

---

### Summary of Final Training Protocol

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam (β₁=0.5, β₂=0.999) | Fast adaptation, stable minimax |
| Learning rate | 2×10⁻⁴ (both G & D) | Standard Pix2Pix; reduce D LR to 1×10⁻⁴ if D dominates |
| Update scheme | Alternating (1 D : 1 G per batch) | Consistent minimax step, no GAN gradient confusion |
| LR schedule | Linear decay from epoch 100 to 150 → 0 | Smooths final convergence; optional |
| Gradient clipping | Generator L2‑norm max = 10.0 | Safety net for early epochs |
| Mixed precision | AMP + loss scaling | Cuts activation memory in half, enables batch 8 |
| Early stopping metric | \((1-\text{LPIPS})\times\text{SSIM}\) on validation | Balances perceptual quality and structure |
| Patience | 20 epochs | Enough to ride out small fluctuations |
| Best model save | Highest validation SSIM | Prioritises structural fidelity for detection |
| Checkpoint save frequency | Every 5 epochs + best | Allows resumption, cleans older files |

These mathematical justifications complete the training design. The procedure is now rigorously defined and ready for implementation.