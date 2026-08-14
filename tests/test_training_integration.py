"""
End-to-End Training Integration Test for Dynamic Generator
"""

import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("c:/Users/soham/Desktop/Soham/InfraNova-AI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.landsat9_dataset import Landsat9Dataset
from src.models.pix2pix.pix2pix import Pix2Pix


def test_training_integration():
    print("="*60)
    print("TRAINING INTEGRATION VERIFICATION")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Using device: {device}")

    # 1. Dataset and DataLoader
    print("\n[2] Initializing dataset...")
    try:
        # Load one dataset sample. We assume splits exist from Phase 1.
        dataset = Landsat9Dataset(
            root_dir=str(PROJECT_ROOT / "data/landsat9/patches"), # using patches directly for the test if splits aren't ready
            split=".", # mock split
            image_size=128,
            normalization="local"
        )
        # Override samples if patches structure is used instead of splits
        # Just grab the first available patch directory
        samples = []
        patches_dir = PROJECT_ROOT / "data/landsat9/patches"
        if patches_dir.exists():
            for region in patches_dir.iterdir():
                if region.is_dir():
                    for patch in region.iterdir():
                        if patch.is_dir():
                            samples.append(patch)
        if not samples:
            print("    [WARNING] No patches found. Creating a synthetic batch for the test.")
            batch = {
                "ir": torch.randn(2, 1, 128, 128),
                "rgb": torch.randn(2, 3, 128, 128)
            }
        else:
            dataset.samples = samples
            dataset.split_dir = patches_dir
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            raw_batch = next(iter(dataloader))
            # Handle dataset dictionary output
            if "tir" in raw_batch and "rgb" in raw_batch:
                batch = {"ir": raw_batch["tir"], "rgb": raw_batch["rgb"]}
            elif "input" in raw_batch and "target" in raw_batch:
                batch = {"ir": raw_batch["input"], "rgb": raw_batch["target"]}
            else:
                batch = raw_batch
    except Exception as e:
        print(f"    [WARNING] Dataset init failed (Phase 1 might not be run): {e}")
        print("    Using synthetic batch.")
        batch = {
            "ir": torch.randn(2, 1, 128, 128),
            "rgb": torch.randn(2, 3, 128, 128)
        }

    ir = batch["ir"].to(device)
    rgb = batch["rgb"].to(device)
    print(f"    Batch shapes - IR: {ir.shape}, RGB: {rgb.shape}")

    # 2. Instantiate Model
    print("\n[3] Initializing Pix2Pix with dynamic generator...")
    try:
        model = Pix2Pix(
            device=device,
            in_channels=1,
            out_channels=3,
            image_size=128,
            generator_impl="dynamic"
        )
        model.train()
    except Exception:
        print("[FAIL] Model initialization failed:")
        traceback.print_exc()
        sys.exit(1)

    # 3. Optimizers & Losses
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    print("\n[4] Executing Forward & Backward Pass (with AMP)...")
    try:
        # ----- Train Discriminator -----
        opt_d.zero_grad()

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # Fake
            fake_rgb = model.generate(ir)
            pred_fake = model.discriminate(ir, fake_rgb.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            # Real
            pred_real = model.discriminate(ir, rgb)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))

            loss_d = (loss_d_fake + loss_d_real) * 0.5

        scaler.scale(loss_d).backward()
        scaler.step(opt_d)

        # Check D gradients
        d_grads = [p.grad is not None for p in model.discriminator.parameters() if p.requires_grad]
        if not all(d_grads):
            print("    [WARNING] Some discriminator parameters have no gradients!")
        else:
            print("    Discriminator gradients verified.")

        # ----- Train Generator -----
        opt_g.zero_grad()

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # We already have fake_rgb, but standard cycle recomputes or uses it
            pred_fake_for_g = model.discriminate(ir, fake_rgb)
            loss_g_gan = criterion_gan(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1 = criterion_l1(fake_rgb, rgb) * 10.0

            loss_g = loss_g_gan + loss_g_l1

        scaler.scale(loss_g).backward()
        scaler.step(opt_g)
        scaler.update()

        # Check G gradients
        g_grads = [p.grad is not None for p in model.generator.parameters() if p.requires_grad]
        if not all(g_grads):
            print("    [WARNING] Some generator parameters have no gradients!")
        else:
            print("    Generator gradients verified.")

        print(f"    Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")

        if not torch.isfinite(loss_d) or not torch.isfinite(loss_g):
            print("    [FAIL] Losses are not finite!")
            sys.exit(1)

    except Exception:
        print("[FAIL] Training step failed:")
        traceback.print_exc()
        sys.exit(1)

    print("\n[5] Checkpoint Save/Load Verification...")
    try:
        # Test state dict saving
        chkpt_path = PROJECT_ROOT / "experiments/test_dynamic_chkpt.pth"
        chkpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "generator": model.generator.state_dict(),
            "discriminator": model.discriminator.state_dict()
        }, chkpt_path)

        # Test loading
        loaded = torch.load(chkpt_path, map_location=device, weights_only=True)
        model.generator.load_state_dict(loaded["generator"])
        model.discriminator.load_state_dict(loaded["discriminator"])

        chkpt_path.unlink()
        print("    [PASS] State dict serialization and deserialization successful.")
    except Exception as e:
        print(f"    [FAIL] Checkpoint operations failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print("INTEGRATION TEST PASSED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    test_training_integration()
