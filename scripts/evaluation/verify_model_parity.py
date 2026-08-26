import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.inference import InferenceEngine
from demo.utils import preprocess_ir_image

CKPT_PATH = PROJECT_ROOT / "outputs" / "best" / "pix2pix_landsat_best.pth"
TEST_PATCH_DIR = PROJECT_ROOT / "data" / "landsat9_b10_b11" / "splits" / "test" / "adilabad_sample_000_12262"
BASELINE_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "parity_baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 using buffered streaming to avoid loading large files into memory."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def record_baseline():
    print("=" * 80)
    print("RECORDING PRE-CLEANUP BASELINE (STREAMING HASH & MODEL METADATA)")
    print("=" * 80)

    ckpt_sha = compute_file_sha256(CKPT_PATH)
    ckpt_size = CKPT_PATH.stat().st_size
    print(f"Checkpoint Path: {CKPT_PATH}")
    print(f"Checkpoint SHA256: {ckpt_sha}")
    print(f"Checkpoint Size: {ckpt_size:,} bytes")

    eng = InferenceEngine(str(CKPT_PATH), image_size=128)
    model = eng.load_model()
    gen_cls = type(model.generator).__name__
    disc_cls = type(model.discriminator).__name__
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())

    print(f"Generator Class: {gen_cls} ({gen_params:,} parameters)")
    print(f"Discriminator Class: {disc_cls} ({disc_params:,} parameters)")
    print(f"Device: {eng.device}")

    # 1. Deterministic Synthetic Input (Seed 42)
    np.random.seed(42)
    synth_input = np.random.uniform(280.0, 310.0, (2, 128, 128)).astype(np.float32)

    # 2. Real Landsat 9 Test Patch
    b10 = np.load(TEST_PATCH_DIR / "tir_100m.npy")
    b11 = np.load(TEST_PATCH_DIR / "tir_b11_100m.npy")
    real_input = np.stack([b10, b11], axis=0)

    model.eval()
    with torch.inference_mode():
        t_synth = preprocess_ir_image(synth_input, image_size=128, target_channels=2).to(eng.device)
        out_synth = model.generate(t_synth).cpu().numpy()

        t_real = preprocess_ir_image(real_input, image_size=128, target_channels=2).to(eng.device)
        out_real = model.generate(t_real).cpu().numpy()

    assert np.isfinite(out_synth).all(), "Synthetic output contains non-finite values"
    assert np.isfinite(out_real).all(), "Real test output contains non-finite values"

    np.save(BASELINE_DIR / "synth_out.npy", out_synth)
    np.save(BASELINE_DIR / "real_out.npy", out_real)

    meta = {
        "ckpt_sha256": ckpt_sha,
        "ckpt_size_bytes": ckpt_size,
        "generator_class": gen_cls,
        "generator_parameters": gen_params,
        "discriminator_class": disc_cls,
        "discriminator_parameters": disc_params,
        "synth_mean": float(out_synth.mean()),
        "synth_std": float(out_synth.std()),
        "real_mean": float(out_real.mean()),
        "real_std": float(out_real.std()),
        "input_shape": list(t_synth.shape),
        "output_shape": list(out_synth.shape),
    }
    (BASELINE_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nBaseline metadata and tensors successfully saved to: {BASELINE_DIR}")
    print(f"Synth Out: shape={out_synth.shape}, mean={out_synth.mean():.6f}, std={out_synth.std():.6f}")
    print(f"Real Out:  shape={out_real.shape}, mean={out_real.mean():.6f}, std={out_real.std():.6f}")


def verify_parity():
    print("=" * 80)
    print("VERIFYING POST-CLEANUP NUMERICAL PARITY & MODEL PRESERVATION")
    print("=" * 80)

    # 1. Checkpoint File Integrity Verification
    meta = json.loads((BASELINE_DIR / "metadata.json").read_text())
    post_sha = compute_file_sha256(CKPT_PATH)
    post_size = CKPT_PATH.stat().st_size
    print(f"Pre-Cleanup SHA256:  {meta['ckpt_sha256']}")
    print(f"Post-Cleanup SHA256: {post_sha}")
    assert post_sha == meta["ckpt_sha256"], "CRITICAL: Checkpoint SHA256 changed after cleanup!"
    assert post_size == meta["ckpt_size_bytes"], "CRITICAL: Checkpoint file size changed after cleanup!"
    print("-> Checkpoint File Preservation: 100% MATCH")

    # 2. Model Structure & Parameter Count Verification
    eng = InferenceEngine(str(CKPT_PATH), image_size=128)
    model = eng.load_model()
    gen_cls = type(model.generator).__name__
    disc_cls = type(model.discriminator).__name__
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())

    print(f"\nGenerator:     {gen_cls} ({gen_params:,} parameters)")
    print(f"Discriminator: {disc_cls} ({disc_params:,} parameters)")

    assert gen_cls == meta["generator_class"], f"Generator class mismatch: {gen_cls} vs {meta['generator_class']}"
    assert disc_cls == meta["discriminator_class"], (
        f"Discriminator class mismatch: {disc_cls} vs {meta['discriminator_class']}"
    )
    assert gen_params == meta["generator_parameters"], (
        f"Generator parameter count mismatch: {gen_params} vs {meta['generator_parameters']}"
    )
    assert disc_params == meta["discriminator_parameters"], (
        f"Discriminator parameter count mismatch: {disc_params} vs {meta['discriminator_parameters']}"
    )
    print("-> Model Architecture & Parameter Counts: 100% MATCH")

    # 3. Deterministic Inference Evaluation
    base_synth = np.load(BASELINE_DIR / "synth_out.npy")
    base_real = np.load(BASELINE_DIR / "real_out.npy")

    np.random.seed(42)
    synth_input = np.random.uniform(280.0, 310.0, (2, 128, 128)).astype(np.float32)

    b10 = np.load(TEST_PATCH_DIR / "tir_100m.npy")
    b11 = np.load(TEST_PATCH_DIR / "tir_b11_100m.npy")
    real_input = np.stack([b10, b11], axis=0)

    model.eval()
    with torch.inference_mode():
        t_synth = preprocess_ir_image(synth_input, image_size=128, target_channels=2).to(eng.device)
        out_synth = model.generate(t_synth).cpu().numpy()

        t_real = preprocess_ir_image(real_input, image_size=128, target_channels=2).to(eng.device)
        out_real = model.generate(t_real).cpu().numpy()

    # 4. Shape and Value Integrity Checks
    assert list(t_synth.shape) == meta["input_shape"], f"Input shape mismatch: {t_synth.shape}"
    assert list(out_synth.shape) == meta["output_shape"], f"Output shape mismatch: {out_synth.shape}"
    assert np.isfinite(out_synth).all(), "Synthetic output contains non-finite values!"
    assert np.isfinite(out_real).all(), "Real test output contains non-finite values!"

    synth_diff = float(np.max(np.abs(out_synth - base_synth)))
    real_diff = float(np.max(np.abs(out_real - base_real)))
    print(f"\nMax Absolute Difference (Synthetic Input): {synth_diff:.2e}")
    print(f"Max Absolute Difference (Real Test Input):  {real_diff:.2e}")

    # Check for bitwise equality if possible
    bitwise_synth = np.array_equal(out_synth, base_synth)
    bitwise_real = np.array_equal(out_real, base_real)
    print(f"Exact Bitwise Equality (Synthetic): {bitwise_synth}")
    print(f"Exact Bitwise Equality (Real Patch): {bitwise_real}")

    assert synth_diff < 1e-6, f"Parity mismatch on synthetic input: max diff = {synth_diff}"
    assert real_diff < 1e-6, f"Parity mismatch on real test input: max diff = {real_diff}"

    print("\n" + "=" * 80)
    print("NUMERICAL PARITY CONFIRMED: max absolute difference < 1e-6")
    print("FROZEN PIX2PIXHD SYSTEM PRESERVED WITH 100% INTEGRITY")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_parity()
    else:
        record_baseline()
