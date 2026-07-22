"""
Export InfraNova AI generator to ONNX format for deployment.

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint checkpoints/best/pix2pix_landsat_best.pth --output model.onnx
    python export_onnx.py --opset 17 --input-size 256
    python export_onnx.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_torch_checkpoint


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
    input_size: int = 256,
    in_channels: int = 1,
) -> None:
    """
    Export the generator from a Pix2Pix checkpoint to ONNX.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        output_path: Path for the exported .onnx file.
        opset_version: ONNX opset version.
        input_size: Spatial size of the dummy input (square).
        in_channels: Number of input channels.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = load_torch_checkpoint(checkpoint_path, map_location="cpu")

    # Build model and load weights
    model = Pix2Pix(in_channels=in_channels, out_channels=3)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "generator_state_dict" in ckpt:
        model.generator.load_state_dict(ckpt["generator_state_dict"])
    else:
        # Try loading directly as a state dict
        model.load_state_dict(ckpt, strict=False)

    generator = model.generator
    generator.eval()

    # Create dummy input
    dummy_input = torch.randn(1, in_channels, input_size, input_size)
    print(f"Input shape: {list(dummy_input.shape)}")

    # Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        generator,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["thermal_input"],
        output_names=["rgb_output"],
        dynamic_axes={
            "thermal_input": {0: "batch_size", 2: "height", 3: "width"},
            "rgb_output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    # Verify
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Exported: {output_path} ({file_size_mb:.1f} MB)")

    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: PASSED")
    except ImportError:
        print("Install 'onnx' package to verify the exported model: pip install onnx")
    except Exception as exc:
        print(f"ONNX model verification: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export InfraNova AI generator to ONNX format."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best/pix2pix_landsat_best.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="exports/infraNova_generator.onnx",
        help="Output ONNX file path.",
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
        default=256,
        help="Spatial input size for the dummy tensor (default: 256).",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        help="Number of input channels (default: 1 for thermal).",
    )
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        input_size=args.input_size,
        in_channels=args.in_channels,
    )


if __name__ == "__main__":
    main()
