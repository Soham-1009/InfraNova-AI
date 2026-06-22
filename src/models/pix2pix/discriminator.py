from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class DiscBlock(nn.Module):
    """Discriminator block: Conv2d -> InstanceNorm -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = [
            spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=not use_norm,
                )
            )
        ]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDiscriminator(nn.Module):
    """
    Pix2Pix PatchGAN Discriminator.

    Input:
        [B, 4, 256, 256]  -> IR + RGB concatenated
    Output:
        [B, 1, 30, 30]

    Note:
        To obtain the standard 70x70 PatchGAN receptive field and 30x30 output
        for 256x256 inputs, the first three convs use stride=2 and the last two
        use stride=1.
    """

    def __init__(
        self,
        in_channels: int = 4,
        features: List[int] | None = None,
    ) -> None:
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.initial = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels,
                    features[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block1 = DiscBlock(features[0], features[1], stride=2, use_norm=True)
        self.block2 = DiscBlock(features[1], features[2], stride=2, use_norm=True)
        self.block3 = DiscBlock(features[2], features[3], stride=1, use_norm=True)

        self.final = spectral_norm(
            nn.Conv2d(
                features[3],
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Kaiming initialization for convolution layers."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Concatenated tensor [B, 4, H, W]

        Returns:
            Patch scores [B, 1, 30, 30]
        """
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.final(x)