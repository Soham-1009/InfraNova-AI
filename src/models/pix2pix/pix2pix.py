from __future__ import annotations

import torch
import torch.nn as nn

from .discriminator import MultiScaleDiscriminator, PatchDiscriminator
from .generator_hd import Pix2PixHDGenerator, Pix2PixHDGlobalResNetGenerator


class Pix2Pix(nn.Module):
    """
    Pix2PixHD wrapper module for InfraNova AI.

    Coordinates the dual-band (Band 10 + Band 11) Generator (Pix2PixHDGenerator or
    Pix2PixHDGlobalResNetGenerator) with the multi-scale PatchGAN discriminator
    (MultiScaleDiscriminator).

    Exposes:
        - generator (Pix2PixHDGenerator or Pix2PixHDGlobalResNetGenerator)
        - discriminator (MultiScaleDiscriminator or PatchDiscriminator)
        - generate(ir)
        - discriminate(ir, rgb, return_features=False)
        - count_parameters()
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        in_channels: int = 2,
        out_channels: int = 3,
        image_size: int = 128,
        multi_scale: bool = True,
        generator_impl: str = "hd",
        num_scales: int = 2,
    ) -> None:
        super().__init__()

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.generator_impl = str(generator_impl).lower()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.num_scales = num_scales
        self.multi_scale = multi_scale or (num_scales > 1)

        # Primary Generator selection
        if self.generator_impl in ("resnet", "global_resnet", "hd_resnet"):
            self.generator = Pix2PixHDGlobalResNetGenerator(
                in_channels=in_channels,
                out_channels=out_channels,
                ngf=64,
                n_downsampling=2,
                n_blocks=9,
            )
        elif self.generator_impl == "hd":
            self.generator = Pix2PixHDGenerator(
                in_channels=in_channels,
                out_channels=out_channels,
                image_size=image_size,
            )
        else:
            raise ValueError(
                f"Unknown generator implementation: '{generator_impl}'. Choose 'hd' or 'resnet'."
            )

        disc_in_channels = in_channels + out_channels
        if self.multi_scale:
            self.discriminator = MultiScaleDiscriminator(
                in_channels=disc_in_channels,
                num_scales=num_scales,
            )
        else:
            self.discriminator = PatchDiscriminator(in_channels=disc_in_channels)

        self.to(self.device)

    def to(self, *args, **kwargs) -> Pix2Pix:
        """Move the model and keep the cached device in sync with its parameters."""
        super().to(*args, **kwargs)
        self.device = next(self.generator.parameters()).device
        return self

    def _apply(self, fn, recurse: bool = True) -> Pix2Pix:
        """Keep the public device cache correct for all PyTorch device-moving paths."""
        super()._apply(fn, recurse=recurse)
        self.device = next(self.generator.parameters()).device
        return self

    def _model_device(self) -> torch.device:
        """Return the current generator device, including after `.cpu()` or `.cuda()`."""
        return next(self.generator.parameters()).device

    def generate(self, ir: torch.Tensor) -> torch.Tensor:
        """
        Generate RGB image from IR input.

        Args:
            ir: Tensor [B, C_in, H, W] (e.g. [B, 2, 128, 128])

        Returns:
            fake_rgb: Tensor [B, 3, H, W] in [-1, 1]
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
            ir: Tensor [B, C_in, H, W]
            rgb: Tensor [B, 3, H, W]
            return_features: If True, also return intermediate features for
                feature matching loss.

        Returns:
            Multi-scale: Dict with "scale_0", "scale_1", etc. discriminator outputs.
            Single-scale: Patch score map [B, 1, H', W'] or (scores, features).
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
