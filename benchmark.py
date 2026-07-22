"""
Benchmark InfraNova AI inference performance.

Measures CPU FPS, GPU FPS, average inference time, peak RAM, and peak VRAM.

Usage:
    python benchmark.py
    python benchmark.py --checkpoint checkpoints/best/pix2pix_landsat_best.pth
    python benchmark.py --warmup 5 --iterations 50 --input-size 256
    python benchmark.py --help
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_torch_checkpoint


def get_peak_ram_mb() -> float:
    """Get peak RAM usage in MB (cross-platform)."""
    try:
        import resource
        # Unix
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KB
        if sys.platform == "darwin":
            return peak_kb / (1024 * 1024)
        return peak_kb / 1024
    except ImportError:
        pass
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().peak_wset / (1024 * 1024)
    except (ImportError, AttributeError):
        pass
    return 0.0


@torch.inference_mode()
def benchmark_device(
    generator: torch.nn.Module,
    device: str,
    input_size: int,
    in_channels: int,
    warmup: int,
    iterations: int,
) -> Dict[str, float]:
    """Run benchmark on a specific device."""
    generator = generator.to(device)
    generator.eval()

    dummy = torch.randn(1, in_channels, input_size, input_size, device=device)

    # Warmup
    for _ in range(warmup):
        _ = generator(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # Reset VRAM tracking
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Timed iterations
    times = []
    for _ in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = generator(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    result = {
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "fps": fps,
        "peak_ram_mb": get_peak_ram_mb(),
    }

    if device == "cuda":
        result["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark InfraNova AI inference performance."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best/pix2pix_landsat_best.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of timed iterations (default: 50).",
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

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print("=" * 60)
    print("InfraNova AI — Inference Benchmark")
    print("=" * 60)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Input size:  {args.in_channels} × {args.input_size} × {args.input_size}")
    print(f"Warmup:      {args.warmup} iterations")
    print(f"Timed:       {args.iterations} iterations")
    print(f"PyTorch:     {torch.__version__}")
    print(f"CUDA:        {'available' if torch.cuda.is_available() else 'not available'}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    ckpt = load_torch_checkpoint(args.checkpoint, map_location="cpu")
    model = Pix2Pix(in_channels=args.in_channels, out_channels=3)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt, strict=False)
    generator = model.generator
    generator.eval()
    print("done.\n")

    # CPU benchmark
    print("Benchmarking CPU...")
    cpu_results = benchmark_device(
        generator, "cpu", args.input_size, args.in_channels,
        args.warmup, args.iterations,
    )

    # GPU benchmark
    gpu_results = None
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        print("Benchmarking GPU...")
        gpu_results = benchmark_device(
            generator, "cuda", args.input_size, args.in_channels,
            args.warmup, args.iterations,
        )

    # Results table
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"{'Metric':<25s} {'CPU':>12s}", end="")
    if gpu_results:
        print(f" {'GPU':>12s}", end="")
    print()
    print("-" * 60)

    rows = [
        ("Avg inference (ms)", "avg_time_ms", ".2f"),
        ("Min inference (ms)", "min_time_ms", ".2f"),
        ("Max inference (ms)", "max_time_ms", ".2f"),
        ("FPS", "fps", ".1f"),
        ("Peak RAM (MB)", "peak_ram_mb", ".0f"),
    ]

    for label, key, fmt in rows:
        cpu_val = cpu_results.get(key, 0.0)
        line = f"{label:<25s} {cpu_val:>12{fmt}}"
        if gpu_results:
            gpu_val = gpu_results.get(key, 0.0)
            line += f" {gpu_val:>12{fmt}}"
        print(line)

    if gpu_results and "peak_vram_mb" in gpu_results:
        vram = gpu_results["peak_vram_mb"]
        print(f"{'Peak VRAM (MB)':<25s} {'—':>12s} {vram:>12.0f}")

    # Speedup
    if gpu_results:
        speedup = cpu_results["avg_time_ms"] / max(gpu_results["avg_time_ms"], 1e-6)
        print(f"\nGPU speedup: {speedup:.1f}×")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
