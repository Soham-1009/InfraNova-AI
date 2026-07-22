"""
Generate a model summary for InfraNova AI.

Outputs total parameters, trainable parameters, per-layer breakdown,
estimated FLOPs, and memory footprint.

Usage:
    python model_summary.py
    python model_summary.py --checkpoint checkpoints/best/pix2pix_landsat_best.pth
    python model_summary.py --output model_summary.txt
    python model_summary.py --help
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.pix2pix import Pix2Pix


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def layer_summary(model: torch.nn.Module, prefix: str = "") -> list:
    """Generate per-layer parameter summary."""
    rows = []
    for name, module in model.named_modules():
        if not list(module.children()):  # Leaf modules only
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                dtype = next(module.parameters()).dtype if list(module.parameters()) else "N/A"
                rows.append({
                    "name": name,
                    "type": type(module).__name__,
                    "params": params,
                    "dtype": str(dtype),
                })
    return rows


def estimate_memory_mb(model: torch.nn.Module) -> float:
    """Estimate model memory in MB (parameters only, float32)."""
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 * 1024)


def estimate_flops(model: torch.nn.Module, input_size: tuple) -> str:
    """Estimate FLOPs using thop if available."""
    try:
        from thop import profile
        dummy = torch.randn(*input_size)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        if flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        return f"{flops / 1e6:.2f} MFLOPs"
    except ImportError:
        return "N/A (install 'thop' package)"
    except Exception as exc:
        return f"N/A ({exc})"


def generate_summary(
    in_channels: int = 1,
    input_size: int = 256,
    output_path: str = "model_summary.txt",
) -> str:
    """Generate and return the full model summary text."""
    model = Pix2Pix(in_channels=in_channels, out_channels=3)
    model.eval()

    buf = StringIO()

    def p(text: str = "") -> None:
        buf.write(text + "\n")

    p("=" * 80)
    p("InfraNova AI — Model Summary")
    p("=" * 80)
    p()

    # Overall stats
    gen_params = count_parameters(model.generator)
    disc_params = count_parameters(model.discriminator)
    total_params = count_parameters(model)

    p("Component Parameters")
    p("-" * 50)
    p(f"  {'Generator':<20s} {gen_params['total']:>12,d} params")
    p(f"  {'Discriminator':<20s} {disc_params['total']:>12,d} params")
    p(f"  {'Total':<20s} {total_params['total']:>12,d} params")
    p(f"  {'Trainable':<20s} {total_params['trainable']:>12,d} params")
    p()

    # Memory
    gen_mem = estimate_memory_mb(model.generator)
    disc_mem = estimate_memory_mb(model.discriminator)
    total_mem = estimate_memory_mb(model)
    p("Memory Estimate (FP32)")
    p("-" * 50)
    p(f"  Generator:     {gen_mem:.1f} MB")
    p(f"  Discriminator: {disc_mem:.1f} MB")
    p(f"  Total:         {total_mem:.1f} MB")
    p()

    # FLOPs
    p("Estimated FLOPs")
    p("-" * 50)
    gen_flops = estimate_flops(
        model.generator, (1, in_channels, input_size, input_size)
    )
    p(f"  Generator:     {gen_flops}")
    p()

    # Generator layers
    p("Generator Layer Breakdown")
    p("-" * 80)
    p(f"  {'Layer':<40s} {'Type':<20s} {'Params':>10s}")
    p(f"  {'─' * 40} {'─' * 20} {'─' * 10}")
    gen_layers = layer_summary(model.generator)
    for row in gen_layers:
        name = row["name"][:40]
        p(f"  {name:<40s} {row['type']:<20s} {row['params']:>10,d}")
    p()

    # Discriminator layers
    p("Discriminator Layer Breakdown")
    p("-" * 80)
    p(f"  {'Layer':<40s} {'Type':<20s} {'Params':>10s}")
    p(f"  {'─' * 40} {'─' * 20} {'─' * 10}")
    disc_layers = layer_summary(model.discriminator)
    for row in disc_layers:
        name = row["name"][:40]
        p(f"  {name:<40s} {row['type']:<20s} {row['params']:>10,d}")
    p()

    p("=" * 80)
    p(f"Input:  {in_channels} × {input_size} × {input_size}")
    p(f"Output: 3 × {input_size} × {input_size}")
    p("=" * 80)

    text = buf.getvalue()

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to: {output_path}")

    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate InfraNova AI model summary."
    )
    parser.add_argument(
        "--output",
        default="model_summary.txt",
        help="Output file path (default: model_summary.txt).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Input spatial size (default: 256).",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        help="Number of input channels (default: 1).",
    )
    args = parser.parse_args()

    generate_summary(
        in_channels=args.in_channels,
        input_size=args.input_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
