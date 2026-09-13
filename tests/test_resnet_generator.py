"""
Tests for the faithful Pix2PixHD-style Global ResNet Generator (Exp9 architecture preparation).

Verifies:
1. Generator and block importability
2. Exact parameter counts (11,369,795 total, 11,369,795 trainable, 0 frozen)
3. Forward pass input/output tensor shapes and range
4. Complete gradient flow across all 25 parameter tensors (no dead layers)
5. MultiScaleDiscriminator coupling (5-channel input)
6. Full training loss integration (LSGAN + L1 + Perceptual + SSIM + Chroma + Saturation)
7. Backward compatibility: existing "hd" generator remains untouched as default
8. SaturationLoss numerical stability preservation
9. Config validator validation of generator implementation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.pix2pix import (
    MultiScaleDiscriminator,
    Pix2Pix,
    Pix2PixHDGenerator,
    Pix2PixHDGlobalResNetGenerator,
    ResnetBlock,
)
from src.training.losses import CombinedLoss, SaturationLoss
from src.utils.config_validator import validate_config


def test_imports_and_classes():
    """Verify classes exist and inherit from nn.Module."""
    assert issubclass(ResnetBlock, nn.Module)
    assert issubclass(Pix2PixHDGlobalResNetGenerator, nn.Module)


def test_exact_parameter_counts():
    """
    Verify exact parameter counts match the specification:
    Total: 11,369,795
    Trainable: 11,369,795
    Frozen: 0
    """
    gen = Pix2PixHDGlobalResNetGenerator(
        in_channels=2,
        out_channels=3,
        ngf=64,
        n_downsampling=2,
        n_blocks=9,
    )
    total_params = sum(p.numel() for p in gen.parameters())
    trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in gen.parameters() if not p.requires_grad)

    assert total_params == 11_369_795, f"Expected 11,369,795 total params, got {total_params}"
    assert trainable_params == 11_369_795, f"Expected 11,369,795 trainable params, got {trainable_params}"
    assert frozen_params == 0, f"Expected 0 frozen params, got {frozen_params}"


def test_forward_pass_shapes_and_bounds():
    """Verify forward pass input validation and output shape and range."""
    gen = Pix2PixHDGlobalResNetGenerator(in_channels=2, out_channels=3)
    gen.eval()

    # Valid input
    x = torch.randn(2, 2, 128, 128)
    with torch.no_grad():
        out = gen(x)

    assert out.shape == (2, 3, 128, 128)
    assert out.min() >= -1.0
    assert out.max() <= 1.0

    # Invalid channel count
    with pytest.raises(ValueError, match="Expected 2 input channels"):
        gen(torch.randn(2, 1, 128, 128))

    # Invalid rank
    with pytest.raises(ValueError, match="Expected input shaped"):
        gen(torch.randn(2, 128, 128))


def test_full_gradient_flow():
    """
    Verify every single parameter tensor receives a finite, non-zero gradient.
    Unlike the old generator which discarded coarse RGB and had 0 gradients
    on final_up, the faithful ResNet generator must have 100% gradient flow.
    """
    gen = Pix2PixHDGlobalResNetGenerator(in_channels=2, out_channels=3)
    gen.train()

    x = torch.randn(2, 2, 128, 128)
    out = gen(x)
    loss = out.sum()
    loss.backward()

    param_list = list(gen.named_parameters())
    assert len(param_list) == 25, f"Expected 25 parameter tensors, got {len(param_list)}"

    for name, param in param_list:
        assert param.grad is not None, f"Parameter {name} did not receive a gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"
        grad_norm = param.grad.abs().sum().item()
        assert grad_norm > 0, f"Parameter {name} has zero gradient (dead layer)"


def test_multiscale_discriminator_compatibility():
    """
    Verify coupling with MultiScaleDiscriminator:
    Concatenate 2-channel IR + 3-channel generated RGB -> 5-channel input.
    Both scales (scale_0, scale_1) must process without error.
    """
    gen = Pix2PixHDGlobalResNetGenerator(in_channels=2, out_channels=3)
    disc = MultiScaleDiscriminator(in_channels=5, num_scales=2)

    ir = torch.randn(2, 2, 128, 128)
    fake_rgb = gen(ir)

    d_in = torch.cat([ir, fake_rgb], dim=1)
    assert d_in.shape == (2, 5, 128, 128)

    disc_out = disc(d_in, return_features=True)
    assert "scale_0" in disc_out
    assert "scale_1" in disc_out

    score_0, feats_0 = disc_out["scale_0"]
    score_1, feats_1 = disc_out["scale_1"]

    assert score_0.shape == (2, 1, 14, 14)
    assert score_1.shape == (2, 1, 6, 6)
    assert len(feats_0) == 5
    assert len(feats_1) == 5


def test_full_training_loss_compatibility():
    """
    Verify integration with CombinedLoss using full Exp7/Exp9 loss configuration:
    - gan_mode: lsgan
    - lambda_adv: 1.0
    - lambda_l1: 10.0
    - lambda_perc: 5.0
    - lambda_ssim: 5.0
    - lambda_chroma: 2.0
    - lambda_sat: 0.05
    - lambda_feat: 0.0
    """
    gen = Pix2PixHDGlobalResNetGenerator(in_channels=2, out_channels=3)
    disc = MultiScaleDiscriminator(in_channels=5, num_scales=2)

    loss_fn = CombinedLoss(
        gan_mode="lsgan",
        lambda_adv=1.0,
        lambda_l1=10.0,
        lambda_perc=5.0,
        lambda_ssim=5.0,
        lambda_chroma=2.0,
        lambda_sat=0.05,
        lambda_feat=0.0,
    )

    ir = torch.randn(2, 2, 128, 128)
    real_rgb = torch.randn(2, 3, 128, 128).clamp(-1, 1)

    fake_rgb = gen(ir)
    d_in = torch.cat([ir, fake_rgb], dim=1)
    pred_fake = disc(d_in, return_features=False)

    loss_dict = loss_fn(pred_fake, fake_rgb, real_rgb)
    total_loss = loss_dict["total"]

    assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss.item()}"
    for k, v in loss_dict.items():
        assert torch.isfinite(v), f"Loss term '{k}' is not finite: {v.item()}"

    total_loss.backward()
    for name, param in gen.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), (
            f"Param {name} gradient not finite after CombinedLoss backward"
        )


def test_pix2pix_wrapper_backward_compatibility():
    """
    Verify Pix2Pix wrapper correctly preserves backward compatibility:
    - Default is generator_impl="hd" (21,383,238 params)
    - Explicit generator_impl="hd" builds Pix2PixHDGenerator
    - generator_impl="resnet" builds Pix2PixHDGlobalResNetGenerator (11,369,795 params)
    """
    p2p_default = Pix2Pix()
    assert p2p_default.generator_impl == "hd"
    assert isinstance(p2p_default.generator, Pix2PixHDGenerator)
    assert p2p_default.count_parameters()[0] == 21_383_238

    p2p_hd = Pix2Pix(generator_impl="hd")
    assert isinstance(p2p_hd.generator, Pix2PixHDGenerator)
    assert p2p_hd.count_parameters()[0] == 21_383_238

    p2p_resnet = Pix2Pix(generator_impl="resnet")
    assert p2p_resnet.generator_impl == "resnet"
    assert isinstance(p2p_resnet.generator, Pix2PixHDGlobalResNetGenerator)
    assert p2p_resnet.count_parameters()[0] == 11_369_795

    with pytest.raises(ValueError, match="Unknown generator implementation"):
        Pix2Pix(generator_impl="invalid_impl")


def test_saturation_loss_numerical_stability():
    """Verify SaturationLoss epsilon fix remains fully intact and stable."""
    sat_loss = SaturationLoss()
    # Test identical tensors
    pred = torch.zeros(2, 3, 32, 32)
    target = torch.zeros(2, 3, 32, 32)
    loss = sat_loss(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0

    # Test extreme saturation difference
    pred_sat = torch.ones(2, 3, 32, 32)
    loss2 = sat_loss(pred_sat, target)
    assert torch.isfinite(loss2)
    assert loss2.item() > 0.0


def test_config_validator_generator_implementation(tmp_path):
    """Verify config validator accepts 'resnet' and 'hd', and flags invalid values."""
    base_cfg = {
        "project": {"name": "test_exp"},
        "dataset": {"root_dir": str(tmp_path), "image_size": 128, "input_channels": 2, "output_channels": 3},
        "training": {"epochs": 10, "batch_size": 4, "decay_start_epoch": 5, "patience": 5},
        "paths": {"checkpoints": str(tmp_path / "ckpts"), "logs": str(tmp_path / "logs")},
    }

    # Default (no generator section) passes
    validate_config(base_cfg)

    # Valid hd
    cfg_hd = {**base_cfg, "model": {"generator": {"implementation": "hd"}}}
    validate_config(cfg_hd)

    # Valid resnet
    cfg_resnet = {**base_cfg, "model": {"generator": {"implementation": "resnet"}}}
    validate_config(cfg_resnet)

    # Invalid generator implementation
    cfg_invalid = {**base_cfg, "model": {"generator": {"implementation": "transformer"}}}
    with pytest.raises(ValueError, match=r"model\.generator\.implementation must be 'hd' or 'resnet'"):
        validate_config(cfg_invalid)
