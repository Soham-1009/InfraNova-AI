"""
Export InfraNova AI generator to ONNX and/or TorchScript format.

Usage:
    python export_model.py
    python export_model.py --format both
    python export_model.py --format torchscript --checkpoint outputs/best/pix2pix_landsat_best.pth
    python export_model.py --format onnx --opset 17 --input-size 256
    python export_model.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.checkpoint import load_pix2pix_model


def _load_generator(checkpoint_path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, object]]:
    """Load and return the generator from a checkpoint."""
    model, architecture = load_pix2pix_model(checkpoint_path, device=device)
    return model.generator, architecture


def export_onnx(
    generator: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """Export generator to ONNX format."""
    if dummy_input.ndim != 4 or dummy_input.shape[-2] <= 0 or dummy_input.shape[-1] <= 0:
        raise ValueError("dummy_input must have shape [B, C, H, W] with positive spatial dimensions")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX (opset {opset_version})...")
    torch.onnx.export(
        generator,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["thermal_input"],
        output_names=["rgb_output"],
        dynamic_axes={"thermal_input": {0: "batch_size"}, "rgb_output": {0: "batch_size"}},
    )

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")

    # Verify
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  Verification: PASSED")
    except ImportError:
        print("  Verification: skipped (install 'onnx' package)")
    except Exception as exc:
        print(f"  Verification: FAILED ({exc})")


def export_torchscript(
    generator: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
) -> None:
    """Export generator to TorchScript format via tracing."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("Exporting TorchScript (traced)...")
    traced = torch.jit.trace(generator, dummy_input)

    # Verify traced model produces same output
    with torch.no_grad():
        original_out = generator(dummy_input)
        traced_out = traced(dummy_input)
        max_diff = float((original_out - traced_out).abs().max())
        print(f"  Trace verification: max_diff={max_diff:.2e}")

    traced.save(output_path)
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export InfraNova AI generator to ONNX and/or TorchScript.")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "outputs" / "best" / "pix2pix_landsat_best.pth"),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports",
        help="Output directory for exported models.",
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "torchscript", "both"],
        default="both",
        help="Export format (default: both).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Spatial input size (default: checkpoint image size).",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=None,
        help="Number of input channels (default: checkpoint architecture).",
    )
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    generator, architecture = _load_generator(args.checkpoint)
    in_channels = int(architecture["input_channels"])
    input_size = args.input_size or int(architecture["image_size"])
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if args.in_channels is not None and args.in_channels != in_channels:
        raise ValueError(
            f"Checkpoint requires {in_channels} input channels, but --in-channels={args.in_channels} was requested."
        )

    dummy_input = torch.randn(1, in_channels, input_size, input_size)
    print(f"Input shape: {list(dummy_input.shape)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("onnx", "both"):
        export_onnx(
            generator,
            dummy_input,
            str(output_dir / "infraNova_generator.onnx"),
            opset_version=args.opset,
        )

    if args.format in ("torchscript", "both"):
        export_torchscript(
            generator,
            dummy_input,
            str(output_dir / "infraNova_generator.pt"),
        )

    print(f"\nExport complete. Files in: {output_dir}")


if __name__ == "__main__":
    main()
