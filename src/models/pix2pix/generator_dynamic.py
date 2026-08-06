from __future__ import annotations

import math

import torch
import torch.nn as nn

from .generator import DownBlock, UpBlock


class GeneratorUNetDynamic(nn.Module):
    """
    Configuration-driven Pix2Pix U-Net Generator.
    Automatically scales encoder depth based on image_size.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        image_size: int = 256,
    ) -> None:
        super().__init__()

        depth = int(math.log2(image_size))
        if 2**depth != image_size or image_size < 32:
            raise ValueError(f"image_size must be a power of 2 >= 32 (got {image_size})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.required_divisor = 2 ** depth

        features = [64, 128, 256]
        while len(features) < depth:
            features.append(512)

        # 1. Build Encoder (depth blocks)
        self.downs = nn.ModuleList()
        self.downs.append(DownBlock(in_channels, features[0], use_norm=False))
        for i in range(1, depth):
            # Original code ONLY disabled norm on the final bottleneck
            use_norm = i != depth - 1
            self.downs.append(DownBlock(features[i-1], features[i], use_norm=use_norm))

        # 2. Build Decoder (depth-1 blocks)
        self.ups = nn.ModuleList()
        for i in range(depth - 1):
            if i == 0:
                in_ch = features[depth - 1]
            else:
                in_ch = features[depth - 1 - i] * 2

            out_ch = features[depth - 2 - i]

            # Original code only used dropout on the first 3 up blocks (up1, up2, up3)
            use_dropout = i < 3
            self.ups.append(UpBlock(in_ch, out_ch, use_dropout=use_dropout))

        # 3. Final Output Layer
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            nn.Conv2d(features[0] * 2, out_channels, kernel_size=3, stride=1, padding=1),
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
        if x.ndim != 4:
            raise ValueError(f"Expected input shaped [B, C, H, W], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.size(1)}")

        height, width = x.shape[-2:]
        if (
            height < self.required_divisor
            or width < self.required_divisor
            or height % self.required_divisor != 0
            or width % self.required_divisor != 0
        ):
            raise ValueError(
                "Generator input height and width must both be multiples of "
                f"{self.required_divisor} (received {height}x{width})."
            )

        # Encode
        d_outs = []
        out = x
        for down in self.downs:
            out = down(out)
            d_outs.append(out)

        # Decode (Bottleneck is d_outs[-1])
        out = d_outs[-1]
        for i, up in enumerate(self.ups):
            out = up(out)
            # skip connection concatenation mapping correctly against encoder depth
            skip = d_outs[-(i + 2)]
            out = torch.cat([out, skip], dim=1)

        return self.final_up(out)
