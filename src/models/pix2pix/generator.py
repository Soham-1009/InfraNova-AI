from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Encoder block: Conv2d -> InstanceNorm -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_norm,
            )
        ]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: ConvTranspose2d -> InstanceNorm -> ReLU -> optional Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeneratorUNet(nn.Module):
    """
    Pix2Pix U-Net Generator.

    Input:
        [B, in_channels, 256, 256]
    Output:
        [B, 3, 256, 256]

    Default in_channels is 10 for multispectral input, but the model remains
    backward compatible with the original 1-channel pipeline.
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        features: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512, 512, 512, 512, 512]

        self.down1 = DownBlock(in_channels, features[0], use_norm=False)
        self.down2 = DownBlock(features[0], features[1])
        self.down3 = DownBlock(features[1], features[2])
        self.down4 = DownBlock(features[2], features[3])
        self.down5 = DownBlock(features[3], features[4])
        self.down6 = DownBlock(features[4], features[5])
        self.down7 = DownBlock(features[5], features[6])
        self.down8 = DownBlock(features[6], features[7], use_norm=False)

        self.up1 = UpBlock(features[7], features[6], use_dropout=True)
        self.up2 = UpBlock(features[6] * 2, features[5], use_dropout=True)
        self.up3 = UpBlock(features[5] * 2, features[4], use_dropout=True)
        self.up4 = UpBlock(features[4] * 2, features[3])
        self.up5 = UpBlock(features[3] * 2, features[2])
        self.up6 = UpBlock(features[2] * 2, features[1])
        self.up7 = UpBlock(features[1] * 2, features[0])
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(
                features[0] * 2,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Kaiming initialization for convolution layers."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)

        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)

        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        return self.up8(u7)