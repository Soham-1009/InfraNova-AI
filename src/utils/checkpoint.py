from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from src.models.pix2pix.pix2pix import Pix2Pix

logger = logging.getLogger(__name__)


def load_torch_checkpoint(path: str | Path, map_location: Any = "cpu") -> Any:
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


def _normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return a tensor-only state dict with common DataParallel prefixes removed."""
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("Checkpoint state dict must map string parameter names to tensors.")

        while key.startswith("module."):
            key = key.removeprefix("module.")
        normalized[key.replace(".module.", ".")] = value

    if not normalized:
        raise ValueError("Checkpoint state dict is empty.")
    return normalized


def _extract_pix2pix_weights(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    """Extract a full-model or generator-only Pix2Pix state dict from known checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Pix2Pix checkpoint must be a mapping.")

    if "model_state_dict" in checkpoint:
        state_dict = _normalize_state_dict_keys(checkpoint["model_state_dict"])
        if state_dict and all(key.startswith("generator.") for key in state_dict):
            return ({key.removeprefix("generator."): value for key, value in state_dict.items()}, "generator")
        return state_dict, "model"

    if "generator_state_dict" in checkpoint:
        return _normalize_state_dict_keys(checkpoint["generator_state_dict"]), "generator"

    if "generator" in checkpoint and isinstance(checkpoint["generator"], Mapping):
        generator_state = _normalize_state_dict_keys(checkpoint["generator"])
        if "discriminator" not in checkpoint or not isinstance(checkpoint["discriminator"], Mapping):
            return generator_state, "generator"

        discriminator_state = _normalize_state_dict_keys(checkpoint["discriminator"])
        model_state = {f"generator.{key}": value for key, value in generator_state.items()}
        model_state.update({f"discriminator.{key}": value for key, value in discriminator_state.items()})
        return model_state, "model"

    state_dict = _normalize_state_dict_keys(checkpoint)
    if any(key.startswith("discriminator.") for key in state_dict):
        return state_dict, "model"
    if state_dict and all(key.startswith("generator.") for key in state_dict):
        return ({key.removeprefix("generator."): value for key, value in state_dict.items()}, "generator")
    return state_dict, "generator"


def _generator_state_dict(state_dict: Mapping[str, torch.Tensor], weight_kind: str) -> dict[str, torch.Tensor]:
    if weight_kind == "generator":
        return dict(state_dict)

    generator_state = {
        key.removeprefix("generator."): value
        for key, value in state_dict.items()
        if key.startswith("generator.")
    }
    if not generator_state:
        raise ValueError("Full Pix2Pix checkpoint does not contain generator weights.")
    return generator_state


def _normalize_generator_impl(value: Any) -> str | None:
    implementation = str(value).lower()
    if implementation in {"resnet", "global_resnet", "hd_resnet"}:
        return "resnet"
    if implementation == "hd":
        return "hd"
    return None


def _infer_generator_impl(generator_state: Mapping[str, torch.Tensor]) -> str:
    keys = set(generator_state)
    if any(key.startswith(("global_gen.", "local_enhancer.", "global_generator.")) for key in keys):
        return "hd"
    if "model.1.weight" in keys:
        return "resnet"
    raise ValueError("Unable to determine the Pix2Pix generator implementation from checkpoint weights.")


def _infer_input_channels(generator_state: Mapping[str, torch.Tensor], generator_impl: str) -> int:
    candidates = (
        ("model.1.weight",)
        if generator_impl == "resnet"
        else ("global_gen.downs.0.block.0.weight", "global_generator.model.0.weight")
    )
    for key in candidates:
        tensor = generator_state.get(key)
        if tensor is not None and tensor.ndim >= 2:
            return int(tensor.shape[1])
    raise ValueError("Unable to determine Pix2Pix input channels from checkpoint weights.")


def _infer_num_scales(state_dict: Mapping[str, torch.Tensor]) -> int:
    scale_indices: set[int] = set()
    for key in state_dict:
        prefix = "discriminator.discriminators."
        if key.startswith(prefix):
            remainder = key.removeprefix(prefix)
            index, _, _ = remainder.partition(".")
            if index.isdigit():
                scale_indices.add(int(index))
    return max(scale_indices) + 1 if scale_indices else 2


def load_pix2pix_model(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[Pix2Pix, dict[str, Any]]:
    """Load a complete Pix2Pix model strictly, inferring its architecture from checkpoint metadata or weights.

    Full training checkpoints and generator-only inference checkpoints are both supported. Missing or
    shape-mismatched tensors raise an error instead of leaving randomly initialized model layers in use.
    """
    checkpoint_path = Path(path)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    state_dict, weight_kind = _extract_pix2pix_weights(checkpoint)
    generator_state = _generator_state_dict(state_dict, weight_kind)

    raw_arch_info = checkpoint.get("arch_info", {}) if isinstance(checkpoint, Mapping) else {}
    arch_info = raw_arch_info if isinstance(raw_arch_info, Mapping) else {}

    inferred_impl = _infer_generator_impl(generator_state)
    metadata_impl = _normalize_generator_impl(arch_info.get("generator_impl", ""))
    if metadata_impl is not None and metadata_impl != inferred_impl:
        raise ValueError(
            "Checkpoint architecture metadata conflicts with its generator weights: "
            f"metadata={metadata_impl}, weights={inferred_impl}."
        )
    generator_impl = metadata_impl or inferred_impl

    inferred_channels = _infer_input_channels(generator_state, generator_impl)
    input_channels = int(arch_info.get("input_channels", arch_info.get("in_channels", inferred_channels)))
    if input_channels != inferred_channels:
        raise ValueError(
            "Checkpoint input-channel metadata conflicts with its generator weights: "
            f"metadata={input_channels}, weights={inferred_channels}."
        )

    out_channels = int(arch_info.get("output_channels", arch_info.get("out_channels", 3)))
    image_size = int(arch_info.get("image_size", 128))
    num_scales = int(arch_info.get("discriminator_scales", _infer_num_scales(state_dict)))
    if input_channels <= 0 or out_channels <= 0 or image_size <= 0 or num_scales <= 0:
        raise ValueError("Checkpoint architecture metadata must contain positive channel, image-size, and scale values.")

    discriminator_impl = str(arch_info.get("discriminator_impl", "")).lower()
    multi_scale = discriminator_impl == "multiscale" or num_scales > 1

    from src.models.pix2pix.pix2pix import Pix2Pix

    model = Pix2Pix(
        device=device,
        in_channels=input_channels,
        out_channels=out_channels,
        image_size=image_size,
        multi_scale=multi_scale,
        generator_impl=generator_impl,
        num_scales=num_scales,
    )
    try:
        if weight_kind == "model":
            model.load_state_dict(state_dict, strict=True)
        else:
            model.generator.load_state_dict(generator_state, strict=True)
    except RuntimeError as exc:
        target = "model" if weight_kind == "model" else "generator"
        raise ValueError(f"Checkpoint {checkpoint_path} cannot be loaded strictly into the inferred {target}.") from exc

    model.eval()
    return model, {
        "generator_impl": generator_impl,
        "input_channels": input_channels,
        "output_channels": out_channels,
        "image_size": image_size,
        "discriminator_scales": num_scales,
        "weight_kind": weight_kind,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    epoch: int,
    metrics: dict[str, float],
    path: str,
    scaler: Any = None,
    scheduler: Any | dict[str, Any] = None,
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
        scheduler: optional scheduler or dict of schedulers
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
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

        # Better architecture-agnostic approach
        in_ch = getattr(model, "in_channels", getattr(gen, "in_channels", None))
        out_ch = getattr(model, "out_channels", getattr(gen, "out_channels", None))

        # Git version
        git_version = "unavailable"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_version = result.stdout.strip()
        except Exception:
            pass

        # Config hash (hash the metrics dict as a proxy for config identity)
        config_hash = hashlib.sha256(str(sorted(metrics.items())).encode()).hexdigest()[:12]

        checkpoint["arch_info"] = {
            "family": "pix2pix",
            "model": type(model).__name__,
            "generator": type(gen).__name__ if gen is not None else "unknown",
            "generator_impl": getattr(model, "generator_impl", "unknown"),
            "discriminator": type(disc).__name__ if disc is not None else "unknown",
            "discriminator_impl": "multiscale" if getattr(model, "multi_scale", False) else "patchgan",
            "discriminator_scales": getattr(model.discriminator, "num_scales", 1)
            if hasattr(model, "discriminator")
            else 1,
            "input_channels": in_ch,
            "output_channels": out_ch,
            "image_size": getattr(model, "image_size", 128),
            "git_version": git_version,
            "config_hash": config_hash,
        }
    except Exception:
        pass  # Don't fail checkpoint saving over metadata extraction

    if isinstance(optimizer, dict):
        checkpoint["optimizer_state_dict"] = {name: opt.state_dict() for name, opt in optimizer.items()}
    else:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if scheduler is not None:
        if isinstance(scheduler, dict):
            checkpoint["scheduler_state_dict"] = {name: sched.state_dict() for name, sched in scheduler.items()}
        else:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    scaler: Any = None,
    scheduler: Any | dict[str, Any] = None,
) -> tuple[int, dict[str, float]]:
    """
    Load a full training checkpoint.

    Args:
        path: checkpoint file path
        model: model instance
        optimizer: single optimizer or dict of optimizers
        scaler: optional GradScaler
        scheduler: optional scheduler or dict of schedulers

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
                expected_in = getattr(model, "in_channels", getattr(gen, "in_channels", None))
                saved_in = arch_info.get("input_channels", arch_info.get("in_channels"))
                if expected_in is not None and saved_in is not None and expected_in != saved_in:
                    errors.append(f"Generator in_channels mismatch: checkpoint={saved_in}, model={expected_in}")

                expected_out = getattr(model, "out_channels", getattr(gen, "out_channels", None))
                saved_out = arch_info.get("output_channels", arch_info.get("out_channels"))
                if expected_out is not None and saved_out is not None and expected_out != saved_out:
                    errors.append(f"Generator out_channels mismatch: checkpoint={saved_out}, model={expected_out}")
            except Exception:
                pass  # Don't fail loading over metadata check

        if errors:
            raise ValueError("Checkpoint architecture mismatch:\n  - " + "\n  - ".join(errors))
    else:
        logger.warning(
            "Checkpoint at %s has no arch_info metadata (legacy format). Skipping compatibility check.",
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

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if isinstance(scheduler, dict):
            sched_state = checkpoint["scheduler_state_dict"]
            for name, sched in scheduler.items():
                if name in sched_state:
                    sched.load_state_dict(sched_state[name])
        else:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics", {})

    return epoch, metrics
