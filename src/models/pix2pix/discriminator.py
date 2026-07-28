from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class DiscBlock(nn.Module):
    """
    Discriminator block: SpectralNorm(Conv2d) -> LeakyReLU.

    Note: InstanceNorm is intentionally omitted when spectral normalization
    is used. Combining spectral norm with instance norm causes the instance
    norm to rescale activations, partially undoing the Lipschitz constraint
    that spectral norm enforces. Using spectral norm alone (SNGAN standard)
    provides more stable adversarial training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
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
                    bias=True,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDiscriminator(nn.Module):
    """
    Pix2Pix PatchGAN Discriminator with optional intermediate feature output.

    Input:
        [B, 4, 256, 256]  -> IR + RGB concatenated
    Output (return_features=False):
        [B, 1, 30, 30]
    Output (return_features=True):
        (final_output, [feat_initial, feat_block1, feat_block2, feat_block3, feat_final])

    Note:
        To obtain the standard 70x70 PatchGAN receptive field and 30x30 output
        for 256x256 inputs, the first three convs use stride=2 and the last two
        use stride=1.

        All convolutional layers use spectral normalization for Lipschitz
        constraint (SNGAN). InstanceNorm is NOT used — see DiscBlock docstring.
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

        self.block1 = DiscBlock(features[0], features[1], stride=2)
        self.block2 = DiscBlock(features[1], features[2], stride=2)
        self.block3 = DiscBlock(features[2], features[3], stride=1)

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

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Concatenated tensor [B, 4, H, W]
            return_features: If True, also return intermediate feature maps.

        Returns:
            If return_features is False: Patch scores [B, 1, 30, 30]
            If return_features is True: (patch_scores, [feat0, feat1, feat2, feat3, feat4])
        """
        f0 = self.initial(x)
        f1 = self.block1(f0)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        out = self.final(f3)

        if return_features:
            return out, [f0, f1, f2, f3, out]
        return out


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator.

    Uses two discriminator branches:
    - Fine: operates on the original resolution input
    - Coarse: operates on 2x-downsampled input

    Both branches share the same architecture but have independent weights.
    This enforces realism at both local (fine) and global (coarse) scales.

    Args:
        in_channels: Number of input channels (typically IR + RGB = 4).
        features: Channel widths for the PatchGAN blocks.
    """

    def __init__(
        self,
        in_channels: int = 4,
        features: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.disc_fine = PatchDiscriminator(in_channels=in_channels, features=features)
        self.disc_coarse = PatchDiscriminator(in_channels=in_channels, features=features)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]]:
        """
        Forward pass through both scales.

        Args:
            x: Concatenated IR+RGB [B, C, H, W]
            return_features: Whether to return intermediate feature maps.

        Returns:
            Dictionary with "fine" and "coarse" discriminator outputs.
            Each value is either a tensor or (tensor, [features]) depending on return_features.
        """
        fine_out = self.disc_fine(x, return_features=return_features)
        x_down = self.downsample(x)
        coarse_out = self.disc_coarse(x_down, return_features=return_features)

        return {"fine": fine_out, "coarse": coarse_out}