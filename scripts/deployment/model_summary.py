"""
Generate a model summary for InfraNova AI.

Outputs total parameters, trainable parameters, per-layer breakdown,
estimated FLOPs, and memory footprint.

Usage:
    python model_summary.py
    python model_summary.py --checkpoint outputs/best/pix2pix_landsat_best.pth
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_pix2pix_model


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
                rows.append(
                    {
                        "name": name,
                        "type": type(module).__name__,
                        "params": params,
                        "dtype": str(dtype),
                    }
                )
    return rows


def estimate_memory_mb(model: torch.nn.Module) -> float:
    """Estimate model memory in MB (parameters only, float32)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 * 1024)


def estimate_flops(model: torch.nn.Module, input_size: tuple) -> str:
    """Estimate FLOPs using thop if available."""
    try:
        from thop import profile

        dummy = torch.randn(*input_size)
        flops, _params = profile(model, inputs=(dummy,), verbose=False)
        if flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        return f"{flops / 1e6:.2f} MFLOPs"
    except ImportError:
        return "N/A (install 'thop' package)"
    except Exception as exc:
        return f"N/A ({exc})"


def generate_summary(
    in_channels: int | None = None,
    input_size: int | None = None,
    output_path: str = "model_summary.txt",
    checkpoint_path: str | None = None,
) -> str:
    """Generate and return the full model summary text."""
    if checkpoint_path is None:
        in_channels = 1 if in_channels is None else in_channels
        input_size = 256 if input_size is None else input_size
        model = Pix2Pix(in_channels=in_channels, out_channels=3)
    else:
        model, architecture = load_pix2pix_model(checkpoint_path, device="cpu")
        checkpoint_channels = int(architecture["input_channels"])
        checkpoint_size = int(architecture["image_size"])
        if in_channels is not None and in_channels != checkpoint_channels:
            raise ValueError(
                f"Checkpoint requires {checkpoint_channels} input channels, but {in_channels} was requested."
            )
        in_channels = checkpoint_channels
        input_size = checkpoint_size if input_size is None else input_size

    if input_size <= 0:
        raise ValueError("input_size must be positive")
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
    gen_flops = estimate_flops(model.generator, (1, in_channels, input_size, input_size))
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
    p(f"Input:  {in_channels} x {input_size} x {input_size}")
    p(f"Output: {model.out_channels} x {input_size} x {input_size}")
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
    parser = argparse.ArgumentParser(description="Generate InfraNova AI model summary.")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "outputs" / "best" / "pix2pix_landsat_best.pth"),
        help="Checkpoint used to derive the model architecture.",
    )
    parser.add_argument(
        "--output",
        default="model_summary.txt",
        help="Output file path (default: model_summary.txt).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Input spatial size (default: checkpoint image size).",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=None,
        help="Number of input channels (default: checkpoint architecture).",
    )
    args = parser.parse_args()

    generate_summary(
        in_channels=args.in_channels,
        input_size=args.input_size,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
