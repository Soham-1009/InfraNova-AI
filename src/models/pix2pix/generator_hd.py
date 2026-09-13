"""
Pix2PixHD-inspired Generator
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    """Encoder block: Conv2d -> InstanceNorm -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
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
    """Decoder block: Upsample -> Conv2d -> InstanceNorm -> ReLU -> optional Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Upsample(scale_factor=2.0, mode="bilinear",
                        align_corners=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
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


class GlobalGenerator(nn.Module):
    """
    Global Generator (coarse network) for Pix2PixHD-inspired architecture.
    Uses the Dynamic U-Net structure.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, image_size: int = 64) -> None:
        super().__init__()
        depth = int(math.log2(image_size))
        if 2**depth != image_size or image_size < 32:
            raise ValueError(
                f"image_size must be a power of 2 >= 32 (got {image_size})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        features = [64, 128, 256]
        while len(features) < depth:
            features.append(512)

        self.downs = nn.ModuleList()
        self.downs.append(DownBlock(in_channels, features[0], use_norm=False))
        for i in range(1, depth):
            use_norm = i != depth - 1
            self.downs.append(
                DownBlock(features[i - 1], features[i], use_norm=use_norm))

        self.ups = nn.ModuleList()
        for i in range(depth - 1):
            if i == 0:
                in_ch = features[depth - 1]
            else:
                in_ch = features[depth - 1 - i] * 2

            out_ch = features[depth - 2 - i]
            use_dropout = i < 3
            self.ups.append(UpBlock(in_ch, out_ch, use_dropout=use_dropout))

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear",
                        align_corners=False),
            nn.Conv2d(features[0] * 2, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d_outs = []
        out = x
        for down in self.downs:
            out = down(out)
            d_outs.append(out)

        out = d_outs[-1]
        for i, up in enumerate(self.ups):
            out = up(out)
            skip = d_outs[-(i + 2)]
            out = torch.cat([out, skip], dim=1)

        global_feature = out
        coarse_rgb = self.final_up(out)
        return coarse_rgb, global_feature


class LocalEnhancerBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4,
                      stride=2, padding=1, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalEnhancerUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear",
                        align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalEnhancer(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, global_feat_channels: int = 128) -> None:
        super().__init__()
        self.enc1 = LocalEnhancerBlock(in_channels, 32, use_norm=False)
        self.enc2 = LocalEnhancerBlock(32, 64, use_norm=True)

        self.fusion_proj = nn.Conv2d(global_feat_channels, 64, kernel_size=1)

        self.dec1 = LocalEnhancerUpBlock(64, 32)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear",
                        align_corners=False),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.apply(GlobalGenerator._init_weights)

    def forward(self, x: torch.Tensor, global_feature: torch.Tensor) -> torch.Tensor:
        out = self.enc1(x)
        local_feature = self.enc2(out)

        proj_global = self.fusion_proj(global_feature)
        fused = local_feature + proj_global

        out = self.dec1(fused)
        out = self.final_up(out)
        return out


class Pix2PixHDGenerator(nn.Module):
    """
    Pix2PixHD-inspired generator containing a global U-Net generator
    and a local enhancer network.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, image_size: int = 128) -> None:
        super().__init__()
        if image_size != 128:
            raise ValueError(
                f"Pix2PixHDGenerator is currently constrained to image_size=128 (got {image_size})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        self.global_gen = GlobalGenerator(
            in_channels, out_channels, image_size=image_size // 2)
        self.local_enhancer = LocalEnhancer(
            in_channels, out_channels, global_feat_channels=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shaped [B, C, H, W], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.size(1)}")

        x_down = F.avg_pool2d(x, kernel_size=2, stride=2)
        _, global_feature = self.global_gen(x_down)
        fine_rgb = self.local_enhancer(x, global_feature)

        return fine_rgb


class ResnetBlock(nn.Module):
    """
    Residual block with reflection padding:
    ReflectionPad2d(1) -> Conv2d -> InstanceNorm2d -> ReLU ->
    ReflectionPad2d(1) -> Conv2d -> InstanceNorm2d -> Residual addition.
    """

    def __init__(self, dim: int, use_bias: bool = False) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class Pix2PixHDGlobalResNetGenerator(nn.Module):
    """
    Faithful Pix2PixHD-style Global ResNet Generator (Wang et al. 2018 / Johnson et al. 2016).

    Architecture:
        - 7x7 reflection-padded input conv: 2 -> 64 ch, InstanceNorm2d, ReLU
        - Downsample 1: 3x3 stride-2 conv: 64 -> 128 ch, InstanceNorm2d, ReLU (64x64)
        - Downsample 2: 3x3 stride-2 conv: 128 -> 256 ch, InstanceNorm2d, ReLU (32x32)
        - Residual engine: 9 ResnetBlocks at 256 ch (32x32 spatial resolution)
        - Upsample 1: 3x3 stride-2 transposed conv: 256 -> 128 ch, InstanceNorm2d, ReLU (64x64)
        - Upsample 2: 3x3 stride-2 transposed conv: 128 -> 64 ch, InstanceNorm2d, ReLU (128x128)
        - 7x7 reflection-padded output conv: 64 -> 3 ch, Tanh (128x128)

    Exact parameter count: 11,369,795 (at in_channels=2, out_channels=3, ngf=64, n_downsampling=2, n_blocks=9).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 3,
        ngf: int = 64,
        n_downsampling: int = 2,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling stages (128x128 -> 64x64 -> 32x32)
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # Residual engine (9 blocks at 256 channels, 32x32)
        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, use_bias=False)]

        # Upsampling stages (32x32 -> 64x64 -> 128x128)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True),
            ]

        # Final output block (128x128)
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shaped [B, C, H, W], got {tuple(x.shape)}"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.size(1)}"
            )
        return self.model(x)
