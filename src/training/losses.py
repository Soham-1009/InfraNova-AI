from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19

logger = logging.getLogger(__name__)


class GANLoss(nn.Module):
    """
    GAN loss supporting BCE (original Pix2Pix) and LSGAN (MSE) modes.

    Args:
        mode: "bce" for BCEWithLogitsLoss, "lsgan" for MSELoss.
        label_smoothing: Smoothing for real labels (BCE mode only).
    """

    def __init__(
        self,
        mode: str = "bce",
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.smoothing = label_smoothing

        if self.mode == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.mode == "lsgan":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown GAN loss mode: {mode}. Use 'bce' or 'lsgan'.")

    def forward(
        self,
        predictions: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        if target_is_real:
            if self.mode == "bce":
                target = torch.ones_like(predictions) * (1.0 - self.smoothing)
            else:
                target = torch.ones_like(predictions)
        else:
            if self.mode == "bce":
                target = torch.zeros_like(predictions) + self.smoothing
            else:
                target = torch.zeros_like(predictions)
        return self.criterion(predictions, target)


class PixelL1Loss(nn.Module):
    """Standard L1 loss wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, fake_rgb: torch.Tensor, real_rgb: torch.Tensor) -> torch.Tensor:
        return self.criterion(fake_rgb, real_rgb)


class ChromaLoss(nn.Module):
    """
    Color saturation loss to combat desaturation.

    Measures L1 distance between per-pixel saturation (approximated as
    channel standard deviation) of generated and target images.

    Input tensors expected in [-1, 1].
    """

    def forward(self, fake_rgb: torch.Tensor, real_rgb: torch.Tensor) -> torch.Tensor:
        # Convert to [0, 1]
        fake = (fake_rgb + 1.0) / 2.0
        real = (real_rgb + 1.0) / 2.0

        # Per-pixel saturation ≈ std across RGB channels
        fake_sat = fake.std(dim=1, keepdim=True)
        real_sat = real.std(dim=1, keepdim=True)

        return F.l1_loss(fake_sat, real_sat)


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss from Pix2PixHD.

    Computes L1 distance between discriminator intermediate feature maps
    for real and fake pairs. This provides a richer training signal than
    the binary real/fake adversarial loss alone.

    Args:
        num_layers: Number of feature layers to match.
        weight_per_layer: If True, normalize loss per layer. Otherwise sum all.
    """

    def __init__(self, num_layers: int = 5) -> None:
        super().__init__()
        self.num_layers = num_layers

    def forward(
        self,
        fake_features: list[torch.Tensor],
        real_features: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = fake_features[0].new_tensor(0.0)
        n = min(len(fake_features), len(real_features), self.num_layers)
        for i in range(n):
            loss = loss + F.l1_loss(fake_features[i], real_features[i].detach())
        return loss / max(n, 1)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 feature maps.

    Assumes input tensors are normalized to [-1, 1].
    Converts them internally to ImageNet-normalized space.
    """

    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        if layer_weights is None:
            layer_weights = {
                "relu1_2": 1.0,
                "relu2_2": 1.0,
                "relu3_4": 1.0,
                "relu4_4": 1.0,
            }

        self.layer_weights = layer_weights
        self.enabled = True

        try:
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        except Exception:
            # Fall back to uninitialized weights if pretrained weights are not available.
            # This keeps offline environments and Docker builds from crashing.
            logger.warning("VGG19 pretrained weights unavailable; perceptual loss is disabled.")
            self.enabled = False
            self.blocks = nn.ModuleList()
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            return

        self.blocks = nn.ModuleList(
            [
                vgg[:4],    # relu1_2
                vgg[4:9],   # relu2_2
                vgg[9:18],  # relu3_4
                vgg[18:27], # relu4_4
            ]
        )

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = requires_grad

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        # This is the fix: cast the mean and std to match the input image's dtype
        mean = self.mean.to(dtype=x.dtype)
        std = self.std.to(dtype=x.dtype)
        return (x - mean) / std

    def forward(self, fake_rgb: torch.Tensor, real_rgb: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return fake_rgb.new_tensor(0.0)

        fake_rgb = self._imagenet_norm(fake_rgb)
        real_rgb = self._imagenet_norm(real_rgb)

        loss = fake_rgb.new_tensor(0.0)

        current_fake = fake_rgb
        current_real = real_rgb

        layer_names = ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]

        for idx, block in enumerate(self.blocks):
            current_fake = block(current_fake)
            current_real = block(current_real)

            layer_name = layer_names[idx]
            weight = self.layer_weights.get(layer_name, 1.0)
            loss = loss + weight * F.l1_loss(current_fake, current_real)

        return loss


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss.

    Input tensors are expected in [-1, 1].
    Loss returned is: 1 - SSIM
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 3
        self.register_buffer("window", self._create_window(window_size, sigma, self.channel))

    @staticmethod
    def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, sigma: float, channel: int) -> torch.Tensor:
        _1d = self._gaussian(window_size, sigma).unsqueeze(1)
        _2d = _1d @ _1d.t()
        window = _2d.unsqueeze(0).unsqueeze(0)
        return window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1 = (0.01 ** 2)
        c2 = (0.03 ** 2)

        channel = img1.size(1)
        # This is the fix: we make sure the 'dtype' matches so it doesn't crash
        if channel != self.channel or self.window.device != img1.device or self.window.dtype != img1.dtype:
            window = self._create_window(self.window_size, self.sigma, channel).to(device=img1.device, dtype=img1.dtype)
            self.window = window
        else:
            window = self.window

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )

        return ssim_map.mean()

    def forward(self, fake_rgb: torch.Tensor, real_rgb: torch.Tensor) -> torch.Tensor:
        fake_rgb = (fake_rgb + 1.0) / 2.0
        real_rgb = (real_rgb + 1.0) / 2.0
        ssim_value = self._ssim(fake_rgb, real_rgb)
        return 1.0 - ssim_value


class CombinedLoss(nn.Module):
    """
    Combined Pix2Pix loss.

    Total = λ_adv * GAN + λ_l1 * L1 + λ_perc * Perceptual
            + λ_ssim * SSIM + λ_chroma * Chroma + λ_feat * FeatureMatching

    Args:
        lambda_adv: Adversarial loss weight.
        lambda_l1: Pixel L1 loss weight.
        lambda_perc: VGG perceptual loss weight.
        lambda_ssim: SSIM loss weight.
        lambda_chroma: Chroma/saturation loss weight.
        lambda_feat: Feature matching loss weight. Set to 0 to disable.
        gan_mode: "bce" or "lsgan".
    """

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_l1: float = 10.0,
        lambda_perc: float = 10.0,
        lambda_ssim: float = 5.0,
        lambda_chroma: float = 2.0,
        lambda_feat: float = 0.0,
        gan_mode: str = "bce",
    ) -> None:
        super().__init__()

        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.lambda_chroma = lambda_chroma
        self.lambda_feat = lambda_feat

        self.gan_loss = GANLoss(mode=gan_mode)
        self.l1_loss = PixelL1Loss()
        self.perc_loss = VGGPerceptualLoss() if self.lambda_perc > 0 else None
        self.ssim_loss = SSIMLoss()
        self.chroma_loss = ChromaLoss() if self.lambda_chroma > 0 else None
        self.feat_loss = FeatureMatchingLoss() if self.lambda_feat > 0 else None

    def forward(
        self,
        disc_fake_pred: torch.Tensor,
        fake_rgb: torch.Tensor,
        real_rgb: torch.Tensor,
        fake_features: list[torch.Tensor] | None = None,
        real_features: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute weighted loss components.

        Args:
            disc_fake_pred: discriminator output for fake pairs
            fake_rgb: generated RGB image
            real_rgb: target RGB image
            fake_features: intermediate feature maps from D(fake) (optional)
            real_features: intermediate feature maps from D(real) (optional)

        Returns:
            Dictionary with individual and total losses.
        """
        adv = self.gan_loss(disc_fake_pred, True)
        l1 = self.l1_loss(fake_rgb, real_rgb)
        if self.perc_loss is not None and fake_rgb.size(1) == 3 and real_rgb.size(1) == 3:
            perc = self.perc_loss(fake_rgb, real_rgb)
        else:
            perc = fake_rgb.new_tensor(0.0)
        ssim = self.ssim_loss(fake_rgb, real_rgb)

        if self.chroma_loss is not None and fake_rgb.size(1) == 3 and real_rgb.size(1) == 3:
            chroma = self.chroma_loss(fake_rgb, real_rgb)
        else:
            chroma = fake_rgb.new_tensor(0.0)

        if (
            self.feat_loss is not None
            and fake_features is not None
            and real_features is not None
        ):
            feat = self.feat_loss(fake_features, real_features)
        else:
            feat = fake_rgb.new_tensor(0.0)

        total = (
            self.lambda_adv * adv
            + self.lambda_l1 * l1
            + self.lambda_perc * perc
            + self.lambda_ssim * ssim
            + self.lambda_chroma * chroma
            + self.lambda_feat * feat
        )

        return {
            "total": total,
            "adv": adv,
            "l1": l1,
            "perc": perc,
            "ssim": ssim,
            "chroma": chroma,
            "feat": feat,
        }


# ---------------------------------------------------------------------------
# Standalone color-quality metrics (used by evaluate.py and trainer)
# ---------------------------------------------------------------------------

def compute_mean_saturation_ratio(
    pred: torch.Tensor, target: torch.Tensor,
) -> float:
    """
    Compute ratio of mean saturation (channel std) between prediction and target.

    Inputs: [B, 3, H, W] in [0, 1].
    Returns ratio: 1.0 means identical saturation, < 1.0 means desaturated.
    """
    pred_sat = pred.std(dim=1).mean()
    target_sat = target.std(dim=1).mean()
    if target_sat < 1e-8:
        return 1.0
    return float((pred_sat / target_sat).item())


def compute_color_histogram_distance(
    pred: torch.Tensor, target: torch.Tensor, bins: int = 64,
) -> float:
    """
    Compute chi-squared distance between RGB histograms.

    Inputs: [B, 3, H, W] in [0, 1].
    Returns distance (lower is better, 0 = identical).
    """
    total_dist = 0.0
    for c in range(3):
        p = pred[:, c].flatten()
        t = target[:, c].flatten()
        hist_p = torch.histc(p, bins=bins, min=0.0, max=1.0)
        hist_t = torch.histc(t, bins=bins, min=0.0, max=1.0)
        # Normalize to probability distributions
        hist_p = hist_p / hist_p.sum().clamp_min(1e-8)
        hist_t = hist_t / hist_t.sum().clamp_min(1e-8)
        # Chi-squared distance
        denom = (hist_p + hist_t).clamp_min(1e-8)
        chi2 = ((hist_p - hist_t) ** 2 / denom).sum()
        total_dist += float(chi2.item())
    return total_dist / 3.0


def compute_lab_color_error(
    pred: torch.Tensor, target: torch.Tensor,
) -> float:
    """
    Compute mean CIE Lab color error (Delta E approximation).

    Uses a simplified sRGB->Lab conversion. Inputs: [B, 3, H, W] in [0, 1].
    Returns mean Delta E (lower is better).
    """
    def _srgb_to_lab_approx(rgb: torch.Tensor) -> torch.Tensor:
        """Simplified sRGB -> Lab via linearization + XYZ -> Lab."""
        # Linearize sRGB
        linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
        # sRGB -> XYZ (D65)
        x = 0.4124 * r + 0.3576 * g + 0.1805 * b
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        z = 0.0193 * r + 0.1192 * g + 0.9505 * b
        # Normalize to D65 white point
        x = x / 0.95047
        z = z / 1.08883
        # XYZ -> Lab
        eps = 0.008856
        kappa = 903.3

        def f(t: torch.Tensor) -> torch.Tensor:
            return torch.where(t > eps, t.clamp_min(1e-10).pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

        fx, fy, fz = f(x), f(y), f(z)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b_ch = 200.0 * (fy - fz)
        return torch.cat([L, a, b_ch], dim=1)

    lab_pred = _srgb_to_lab_approx(pred)
    lab_target = _srgb_to_lab_approx(target)
    delta = (lab_pred - lab_target) ** 2
    delta_e = delta.sum(dim=1).sqrt().mean()
    return float(delta_e.item())
