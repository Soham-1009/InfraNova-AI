"""
Configuration validator for InfraNova AI training pipeline.

Validates config.yaml at training startup to catch misconfigurations
before any expensive GPU work begins.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """
    Validate a training configuration dictionary.

    Args:
        cfg: Parsed config.yaml contents.

    Returns:
        List of error messages. Empty list means valid.

    Raises:
        ValueError: If any validation errors are found.
    """
    errors: List[str] = []

    # --- Required top-level keys ---
    required_sections = ("project", "dataset", "training", "paths")
    for section in required_sections:
        if section not in cfg:
            errors.append(f"Missing required config section: '{section}'")

    if errors:
        # Can't validate further without required sections
        raise ValueError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )

    # --- Project ---
    project = cfg.get("project", {})
    if not project.get("name"):
        errors.append("project.name is required")

    seed = project.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append(f"project.seed must be an integer, got {type(seed).__name__}")

    # --- Dataset ---
    dataset = cfg.get("dataset", {})

    image_size = dataset.get("image_size", 256)
    if not isinstance(image_size, int) or image_size <= 0:
        errors.append(f"dataset.image_size must be a positive integer, got {image_size}")

    input_channels = dataset.get("input_channels", 1)
    if not isinstance(input_channels, int) or input_channels <= 0:
        errors.append(f"dataset.input_channels must be a positive integer, got {input_channels}")

    output_channels = dataset.get("output_channels", 3)
    if not isinstance(output_channels, int) or output_channels <= 0:
        errors.append(f"dataset.output_channels must be a positive integer, got {output_channels}")

    root_dir = dataset.get("root_dir", "")
    if root_dir:
        root_path = Path(root_dir)
        if not root_path.exists():
            errors.append(f"dataset.root_dir does not exist: {root_path}")
    else:
        errors.append("dataset.root_dir is required")

    num_workers = dataset.get("num_workers", 0)
    if not isinstance(num_workers, int) or num_workers < 0:
        errors.append(f"dataset.num_workers must be a non-negative integer, got {num_workers}")

    # --- Training ---
    training = cfg.get("training", {})

    epochs = training.get("epochs", 100)
    if not isinstance(epochs, int) or epochs <= 0:
        errors.append(f"training.epochs must be a positive integer, got {epochs}")

    batch_size = training.get("batch_size", 8)
    if not isinstance(batch_size, int) or batch_size <= 0:
        errors.append(f"training.batch_size must be a positive integer, got {batch_size}")

    decay_start = training.get("decay_start_epoch", 100)
    if not isinstance(decay_start, int) or decay_start < 0:
        errors.append(f"training.decay_start_epoch must be a non-negative integer, got {decay_start}")
    elif isinstance(epochs, int) and decay_start > epochs:
        errors.append(
            f"training.decay_start_epoch ({decay_start}) exceeds training.epochs ({epochs})"
        )

    patience = training.get("patience", 20)
    if not isinstance(patience, int) or patience <= 0:
        errors.append(f"training.patience must be a positive integer, got {patience}")

    grad_clip = training.get("grad_clip")
    if grad_clip is not None:
        if not isinstance(grad_clip, (int, float)) or grad_clip <= 0:
            errors.append(f"training.grad_clip must be a positive number, got {grad_clip}")

    # Optimizer
    optim = training.get("optimizer", {})
    lr = optim.get("lr", 2e-4)
    if not isinstance(lr, (int, float)) or lr <= 0:
        errors.append(f"training.optimizer.lr must be a positive number, got {lr}")

    beta1 = optim.get("beta1", 0.5)
    if not isinstance(beta1, (int, float)) or not (0.0 <= beta1 < 1.0):
        errors.append(f"training.optimizer.beta1 must be in [0, 1), got {beta1}")

    beta2 = optim.get("beta2", 0.999)
    if not isinstance(beta2, (int, float)) or not (0.0 <= beta2 < 1.0):
        errors.append(f"training.optimizer.beta2 must be in [0, 1), got {beta2}")

    # Loss weights
    loss_cfg = training.get("loss", {})
    for key in ("lambda_adv", "lambda_l1", "lambda_perc", "lambda_ssim"):
        val = loss_cfg.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val < 0):
            errors.append(f"training.loss.{key} must be a non-negative number, got {val}")

    # Resume checkpoint
    resume_from = training.get("resume_from", "")
    if resume_from and not Path(resume_from).exists():
        logger.warning("training.resume_from points to non-existent file: %s", resume_from)

    # --- Paths ---
    paths = cfg.get("paths", {})
    required_path_keys = ("checkpoints", "logs")
    for key in required_path_keys:
        if key not in paths:
            errors.append(f"paths.{key} is required")

    # Verify checkpoint paths have valid parent directories
    for key in ("best_checkpoint", "latest_checkpoint", "final_checkpoint"):
        ckpt_path = paths.get(key, "")
        if ckpt_path:
            parent = Path(ckpt_path).parent
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                errors.append(f"paths.{key} parent directory cannot be created: {exc}")

    # --- Report ---
    if errors:
        raise ValueError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )

    logger.info("Configuration validation passed.")
    return errors
