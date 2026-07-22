from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import logging
import torch

logger = logging.getLogger(__name__)


def load_torch_checkpoint(path: Union[str, Path], map_location: Any = "cpu") -> Any:
    """
    Load a PyTorch checkpoint using the safer weights-only path when available.

    PyTorch 2.4+ warns when `weights_only` is omitted. The fallback keeps older
    PyTorch versions and legacy checkpoint files usable.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scaler: Any = None,
) -> None:
    """
    Save a full training checkpoint.

    Args:
        model: model to save
        optimizer: single optimizer or dict of optimizers
        epoch: current epoch
        metrics: metric dictionary
        path: file path
        scaler: optional GradScaler
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }

    # Save architecture metadata for compatibility checks on resume
    try:
        import hashlib
        import subprocess

        gen = getattr(model, "generator", None)
        disc = getattr(model, "discriminator", None)

        in_ch = None
        out_ch = None
        if gen is not None:
            first_conv = gen.down1.block[0]
            in_ch = first_conv.in_channels if hasattr(first_conv, "in_channels") else None
            final_layers = list(gen.up8.children()) if hasattr(gen, "up8") else []
            for layer in reversed(final_layers):
                if hasattr(layer, "out_channels"):
                    out_ch = layer.out_channels
                    break

        # Git version
        git_version = "unavailable"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                git_version = result.stdout.strip()
        except Exception:
            pass

        # Config hash (hash the metrics dict as a proxy for config identity)
        config_hash = hashlib.sha256(
            str(sorted(metrics.items())).encode()
        ).hexdigest()[:12]

        checkpoint["arch_info"] = {
            "model": type(model).__name__,
            "generator": type(gen).__name__ if gen is not None else "unknown",
            "discriminator": type(disc).__name__ if disc is not None else "unknown",
            "input_channels": in_ch,
            "output_channels": out_ch,
            "image_size": 256,  # from config default
            "git_version": git_version,
            "config_hash": config_hash,
        }
    except Exception:
        pass  # Don't fail checkpoint saving over metadata extraction

    if isinstance(optimizer, dict):
        checkpoint["optimizer_state_dict"] = {
            name: opt.state_dict() for name, opt in optimizer.items()
        }
    else:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    scaler: Any = None,
) -> Tuple[int, Dict[str, float]]:
    """
    Load a full training checkpoint.

    Args:
        path: checkpoint file path
        model: model instance
        optimizer: single optimizer or dict of optimizers
        scaler: optional GradScaler

    Returns:
        (epoch, metrics)
    """
    checkpoint = load_torch_checkpoint(path, map_location="cpu")

    # Verify architecture compatibility if metadata is available
    arch_info = checkpoint.get("arch_info")
    if arch_info is not None:
        logger.info(
            "Checkpoint arch_info: model=%s, generator=%s, git=%s",
            arch_info.get("model", "?"),
            arch_info.get("generator", "?"),
            arch_info.get("git_version", "?"),
        )
        errors = []
        gen = getattr(model, "generator", None)
        if gen is not None:
            try:
                first_conv = gen.down1.block[0]
                expected_in = first_conv.in_channels if hasattr(first_conv, "in_channels") else None
                # Support both old ("in_channels") and new ("input_channels") key names
                saved_in = arch_info.get("input_channels", arch_info.get("in_channels"))
                if expected_in is not None and saved_in is not None and expected_in != saved_in:
                    errors.append(
                        f"Generator in_channels mismatch: checkpoint={saved_in}, model={expected_in}"
                    )

                final_layers = list(gen.up8.children()) if hasattr(gen, "up8") else []
                expected_out = None
                for layer in reversed(final_layers):
                    if hasattr(layer, "out_channels"):
                        expected_out = layer.out_channels
                        break
                saved_out = arch_info.get("output_channels", arch_info.get("out_channels"))
                if expected_out is not None and saved_out is not None and expected_out != saved_out:
                    errors.append(
                        f"Generator out_channels mismatch: checkpoint={saved_out}, model={expected_out}"
                    )
            except Exception:
                pass  # Don't fail loading over metadata check

        if errors:
            raise ValueError(
                "Checkpoint architecture mismatch:\n  - " + "\n  - ".join(errors)
            )
    else:
        logger.warning(
            "Checkpoint at %s has no arch_info metadata (legacy format). "
            "Skipping compatibility check.",
            path,
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if isinstance(optimizer, dict):
        opt_state = checkpoint.get("optimizer_state_dict", {})
        for name, opt in optimizer.items():
            if name in opt_state:
                opt.load_state_dict(opt_state[name])
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics", {})

    return epoch, metrics
