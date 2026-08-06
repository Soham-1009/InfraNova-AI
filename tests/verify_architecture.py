"""
Architectural Verification and Shape Trace
"""

import sys
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path("c:/Users/soham/Desktop/Soham/InfraNova-AI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.discriminator import PatchDiscriminator
from src.models.pix2pix.generator import GeneratorUNet


def trace_shapes():
    print("="*60)
    print("PIX2PIX ARCHITECTURE SHAPE TRACE")
    print("="*60)

    # 1. Initialize models
    print("\n[1] Initializing GeneratorUNet (8 blocks)...")
    gen = GeneratorUNet(in_channels=1)

    print("\n[2] Initializing PatchDiscriminator...")
    disc = PatchDiscriminator(in_channels=4)

    # 2. Register forward hooks to trace shapes
    shapes = []
    def hook_fn(module, input, output):
        name = module.__class__.__name__
        if hasattr(module, 'trace_name'):
            name = module.trace_name

        in_shape = tuple(input[0].shape)
        if isinstance(output, tuple):
            out_shape = "Tuple: " + str([tuple(o.shape) for o in output])
        else:
            out_shape = tuple(output.shape)

        shapes.append(f"{name:<15} | In: {in_shape!s:<20} | Out: {out_shape!s}")

    # Attach trace names and hooks
    for name, module in gen.named_children():
        module.trace_name = f"Gen:{name}"
        module.register_forward_hook(hook_fn)

    for name, module in disc.named_children():
        module.trace_name = f"Disc:{name}"
        module.register_forward_hook(hook_fn)

    # 3. Forward Pass: Generator
    print("\n[3] Executing Generator Forward Pass (1x1x128x128)...")
    x = torch.randn(1, 1, 128, 128)

    try:
        y = gen(x)
        print("Generator forward pass SUCCEEDED.")
    except Exception as e:
        print("\n!!! GENERATOR CRASHED !!!")
        print("Exception:", type(e).__name__, "-", str(e))
        print("\nTrace:")
        for s in shapes:
            print("  " + s)
        print("\nFull Stack Trace:")
        traceback.print_exc()
        y = None

    if y is not None:
        print("\nGenerator Trace:")
        for s in shapes:
            print("  " + s)

        shapes.clear()

        # 4. Forward Pass: Discriminator
        print("\n[4] Executing Discriminator Forward Pass (1x4x128x128)...")
        # Generator output is 1x3x128x128. Concat with input 1x1x128x128 = 1x4x128x128
        disc_input = torch.cat([x, y], dim=1)
        try:
            _ = disc(disc_input)
            print("Discriminator forward pass SUCCEEDED.")
            print("\nDiscriminator Trace:")
            for s in shapes:
                print("  " + s)
        except Exception as e:
            print("\n!!! DISCRIMINATOR CRASHED !!!")
            print("Exception:", type(e).__name__, "-", str(e))
            traceback.print_exc()

    # 5. Calculate Parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*60)
    print("PARAMETER COUNTS")
    print("="*60)
    print(f"Generator     : {count_parameters(gen):,}")
    print(f"Discriminator : {count_parameters(disc):,}")

    print("\nNow initializing a dynamic GeneratorUNet (Solution D test)...")
    try:
        # Dynamic depth generator logic test (if we were to implement it)
        # For now, just show parameter reduction of 7-blocks
        gen_7 = GeneratorUNet(in_channels=1, features=[64, 128, 256, 512, 512, 512, 512])
        print(f"Gen (7 blocks): {count_parameters(gen_7):,}")
    except Exception as e:
        print(f"Could not init 7-block generator: {e}")

if __name__ == "__main__":
    trace_shapes()
