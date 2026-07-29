from __future__ import annotations

import torch
import torch.nn as nn

from .discriminator import MultiScaleDiscriminator, PatchDiscriminator
from .generator import GeneratorUNet


class Pix2Pix(nn.Module):
    """
    Pix2Pix wrapper module.

    Supports single-scale (PatchGAN) and multi-scale discriminator modes.

    Exposes:
        - generator
        - discriminator
        - generate(ir)
        - discriminate(ir, rgb, return_features=False)
        - count_parameters()
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        in_channels: int = 1,
        out_channels: int = 3,
        multi_scale: bool = False,
    ) -> None:
        super().__init__()

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.multi_scale = multi_scale

        self.generator = GeneratorUNet(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        disc_in_channels = in_channels + out_channels
        if multi_scale:
            self.discriminator = MultiScaleDiscriminator(in_channels=disc_in_channels)
        else:
            self.discriminator = PatchDiscriminator(in_channels=disc_in_channels)

        self.to(self.device)

    def to(self, *args, **kwargs) -> Pix2Pix:
        """Move the model and keep the cached device in sync with its parameters."""
        super().to(*args, **kwargs)
        self.device = next(self.generator.parameters()).device
        return self

    def _model_device(self) -> torch.device:
        """Return the current generator device, including after `.cpu()` or `.cuda()`."""
        return next(self.generator.parameters()).device

    def generate(self, ir: torch.Tensor) -> torch.Tensor:
        """
        Generate RGB image from IR input.

        Args:
            ir: Tensor [B, 1, H, W]

        Returns:
            fake_rgb: Tensor [B, 3, H, W]
        """
        ir = ir.to(self._model_device())
        return self.generator(ir)

    def discriminate(
        self,
        ir: torch.Tensor,
        rgb: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | dict:
        """
        Discriminate concatenated IR + RGB pair.

        Args:
            ir: Tensor [B, 1, H, W]
            rgb: Tensor [B, 3, H, W]
            return_features: If True, also return intermediate features for
                feature matching loss.

        Returns:
            Single-scale: Patch score map [B, 1, 30, 30] or (scores, features)
            Multi-scale: Dict with "fine" and "coarse" outputs
        """
        device = self._model_device()
        ir = ir.to(device)
        rgb = rgb.to(device)
        x = torch.cat([ir, rgb], dim=1)
        return self.discriminator(x, return_features=return_features)

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        """Alias for generate()."""
        return self.generate(ir)

    def count_parameters(self) -> tuple[int, int, int]:
        """
        Count model parameters.

        Returns:
            (generator_params, discriminator_params, total_params)
        """
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        return g_params, d_params, g_params + d_params
