"""
Regression Verification for Dynamic Generator
"""

import sys
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path("c:/Users/soham/Desktop/Soham/InfraNova-AI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.generator import GeneratorUNet
from src.models.pix2pix.generator_dynamic import GeneratorUNetDynamic


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_state_dict_shapes(model):
    return [tuple(t.shape) for t in model.state_dict().values()]

def copy_weights_legacy_to_dynamic(legacy, dynamic):
    """
    Manually copies weights from the old naming scheme (down1, down2)
    to the new naming scheme (downs.0, downs.1) to allow deterministic output comparison.
    """
    leg_state = legacy.state_dict()
    dyn_state = dynamic.state_dict()

    # We rely on the fact that if they are architecturally identical,
    # the sequence of state tensors will match perfectly in order.
    leg_tensors = list(leg_state.values())
    dyn_keys = list(dyn_state.keys())

    assert len(leg_tensors) == len(dyn_keys), "Models have a different number of state tensors!"

    for dyn_key, leg_tensor in zip(dyn_keys, leg_tensors, strict=True):
        dyn_state[dyn_key].copy_(leg_tensor)

    dynamic.load_state_dict(dyn_state)

def run_regression_test():
    print("="*60)
    print("GENERATOR ARCHITECTURE REGRESSION TEST")
    print("="*60)

    # 1. Instantiate both models at 256
    print("\n[1] Instantiating models (in_channels=1, image_size=256)...")
    torch.manual_seed(0)
    gen_old = GeneratorUNet(in_channels=1)
    gen_new = GeneratorUNetDynamic(in_channels=1, image_size=256)

    # 2. Parameter count and Shape Structural verification
    params_old = count_parameters(gen_old)
    params_new = count_parameters(gen_new)
    print(f"Old Parameters : {params_old:,}")
    print(f"New Parameters : {params_new:,}")

    if params_old != params_new:
        print("[FAIL] Parameter counts do not match!")
        sys.exit(1)

    old_shapes = extract_state_dict_shapes(gen_old)
    new_shapes = extract_state_dict_shapes(gen_new)

    if old_shapes != new_shapes:
        print("[FAIL] State dictionary structural shapes do not match!")
        sys.exit(1)
    else:
        print("[PASS] Parameter counts and state dictionary structural shapes match perfectly.")

    # 3. Deterministic Output Comparison
    print("\n[2] Deterministic Numerical Equivalence (Input: 1x1x256x256)...")
    copy_weights_legacy_to_dynamic(gen_old, gen_new)

    # Set to eval mode to disable dropout stochasticity
    gen_old.eval()
    gen_new.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        out_old = gen_old(x)
        out_new = gen_new(x)

    try:
        torch.testing.assert_close(out_old, out_new)
        print("[PASS] Deterministic output numerical equivalence verified!")
    except AssertionError as e:
        print(f"[FAIL] Numerical outputs differ: {e}")
        sys.exit(1)

    # 4. Property-based Scale Testing
    print("\n[3] Property-based Scale Testing...")
    scales = [32, 64, 128, 256, 512]

    for size in scales:
        print(f"  Testing {size}x{size}...")
        try:
            gen_scale = GeneratorUNetDynamic(in_channels=1, image_size=size)
            gen_scale.eval()
            x_scale = torch.randn(1, 1, size, size)
            with torch.no_grad():
                out_scale = gen_scale(x_scale)

            if out_scale.shape[-2:] != (size, size):
                print(f"    [FAIL] Output shape mismatch! Expected {size}, got {out_scale.shape[-2:]}")
                sys.exit(1)
            print(f"    [PASS] Forward pass successful. Output shape: {out_scale.shape}")
        except Exception:
            print(f"    [FAIL] Exception during {size}x{size} test:")
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "="*60)
    print("ALL TESTS PASSED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    run_regression_test()
