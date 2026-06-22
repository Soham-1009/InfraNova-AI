from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .discriminator import PatchDiscriminator
from .generator import GeneratorUNet


class Pix2Pix(nn.Module):
    """
    Pix2Pix wrapper module.

    Exposes:
        - generator
        - discriminator
        - generate(ir)
        - discriminate(ir, rgb)
        - count_parameters()
    """

    def __init__(
        self,
        device: torch.device | str = "cuda",
        in_channels: int = 1,
        out_channels: int = 3,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)

        self.generator = GeneratorUNet(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.discriminator = PatchDiscriminator(
            in_channels=in_channels + out_channels,
        )

        self.to(self.device)

    def generate(self, ir: torch.Tensor) -> torch.Tensor:
        """
        Generate RGB image from IR input.

        Args:
            ir: Tensor [B, 1, H, W]

        Returns:
            fake_rgb: Tensor [B, 3, H, W]
        """
        ir = ir.to(self.device)
        return self.generator(ir)

    def discriminate(self, ir: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        Discriminate concatenated IR + RGB pair.

        Args:
            ir: Tensor [B, 1, H, W]
            rgb: Tensor [B, 3, H, W]

        Returns:
            Patch score map [B, 1, 30, 30]
        """
        ir = ir.to(self.device)
        rgb = rgb.to(self.device)
        x = torch.cat([ir, rgb], dim=1)
        return self.discriminator(x)

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        """Alias for generate()."""
        return self.generate(ir)

    def count_parameters(self) -> Tuple[int, int, int]:
        """
        Count model parameters.

        Returns:
            (generator_params, discriminator_params, total_params)
        """
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        return g_params, d_params, g_params + d_params