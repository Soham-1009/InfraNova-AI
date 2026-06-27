from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19


class GANLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.smoothing = label_smoothing

    def forward(
        self,
        predictions: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        if target_is_real:
            target = torch.ones_like(predictions) * (1.0 - self.smoothing)
        else:
            target = torch.zeros_like(predictions) + self.smoothing
        return self.criterion(predictions, target)


class PixelL1Loss(nn.Module):
    """Standard L1 loss wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, fake_rgb: torch.Tensor, real_rgb: torch.Tensor) -> torch.Tensor:
        return self.criterion(fake_rgb, real_rgb)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 feature maps.

    Assumes input tensors are normalized to [-1, 1].
    Converts them internally to ImageNet-normalized space.
    """

    def __init__(
        self,
        layer_weights: Optional[Dict[str, float]] = None,
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

        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
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
        return (x - self.mean) / self.std

    def forward(self, fake_rgb: torch.Tensor, real_rgb: torch.Tensor) -> torch.Tensor:
        fake_rgb = self._imagenet_norm(fake_rgb)
        real_rgb = self._imagenet_norm(real_rgb)

        loss = torch.tensor(0.0, device=fake_rgb.device)

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
        if channel != self.channel or self.window.device != img1.device:
            window = self._create_window(self.window_size, self.sigma, channel).to(img1.device)
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

    Total = lambda_adv * GAN + lambda_l1 * L1 + lambda_perc * Perceptual + lambda_ssim * SSIM
    """

    def __init__(
        self,
        lambda_adv: float = 0.5,
        lambda_l1: float = 100.0,
        lambda_perc: float = 10.0,
        lambda_ssim: float = 3.0,
    ) -> None:
        super().__init__()

        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim

        self.gan_loss = GANLoss()
        self.l1_loss = PixelL1Loss()
        self.perc_loss = VGGPerceptualLoss()
        self.ssim_loss = SSIMLoss()

    def forward(
        self,
        disc_fake_pred: torch.Tensor,
        fake_rgb: torch.Tensor,
        real_rgb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss components.

        Args:
            disc_fake_pred: discriminator output for fake pairs
            fake_rgb: generated RGB image
            real_rgb: target RGB image

        Returns:
            Dictionary with individual and total losses.
        """
        adv = self.gan_loss(disc_fake_pred, True)
        l1 = self.l1_loss(fake_rgb, real_rgb)
        perc = self.perc_loss(fake_rgb, real_rgb)
        ssim = self.ssim_loss(fake_rgb, real_rgb)

        total = (
            self.lambda_adv * adv
            + self.lambda_l1 * l1
            + self.lambda_perc * perc
            + self.lambda_ssim * ssim
        )

        return {
            "total": total,
            "adv": adv,
            "l1": l1,
            "perc": perc,
            "ssim": ssim,
        }