import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.discriminator import MultiScaleDiscriminator
from src.models.pix2pix.generator_hd import Pix2PixHDGenerator


def test_2channel_generator_forward():
    gen = Pix2PixHDGenerator(in_channels=2, out_channels=3)
    x = torch.randn(2, 2, 128, 128)
    out = gen(x)
    assert out.shape == (2, 3, 128, 128)
    assert torch.isfinite(out).all()


def test_2channel_discriminator_forward():
    # 2 input + 3 output = 5 channels
    disc = MultiScaleDiscriminator(in_channels=5, num_scales=2)
    x_real = torch.randn(2, 5, 128, 128)
    features = disc(x_real)
    assert isinstance(features, dict)
    assert "scale_0" in features and "scale_1" in features
    for _scale, feat_list in features.items():
        assert len(feat_list) > 0
        final_logit = feat_list[-1]
        assert torch.isfinite(final_logit).all()


def test_2channel_one_step_gradients():
    gen = Pix2PixHDGenerator(in_channels=2, out_channels=3)
    disc = MultiScaleDiscriminator(in_channels=5, num_scales=2)

    opt_g = torch.optim.Adam(gen.parameters(), lr=0.0002)
    opt_d = torch.optim.Adam(disc.parameters(), lr=0.0002)

    x = torch.randn(2, 2, 128, 128)
    y = torch.randn(2, 3, 128, 128)

    # Forward G
    fake_y = gen(x)

    # Train D
    opt_d.zero_grad()
    d_real = disc(torch.cat([x, y], dim=1))
    d_fake = disc(torch.cat([x, fake_y.detach()], dim=1))

    d_loss = 0.0
    for scale in d_real:
        d_loss += torch.mean((d_real[scale][-1] - 1.0) ** 2) + torch.mean(d_fake[scale][-1] ** 2)
    d_loss.backward()
    opt_d.step()

    # Train G
    opt_g.zero_grad()
    d_fake_for_g = disc(torch.cat([x, fake_y], dim=1))
    g_loss = torch.mean((fake_y - y).abs())
    for scale in d_fake_for_g:
        g_loss += torch.mean((d_fake_for_g[scale][-1] - 1.0) ** 2)
    g_loss.backward()
    opt_g.step()

    # Verify gradients on parameters that participated in the forward pass
    active_grad_count = 0
    for name, p in gen.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad on G param {name}"
            active_grad_count += 1

    assert active_grad_count > 0, "No gradients accumulated on generator!"

    for name, p in disc.named_parameters():
        assert p.grad is not None, f"No grad on D param {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad on {name}"

    print(f"✅ 2-Channel One-Step Optimization Diagnostic Passed! (Active G params: {active_grad_count})")
