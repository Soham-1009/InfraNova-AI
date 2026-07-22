"""
Reproducibility utilities for InfraNova AI.

Sets all random seeds for deterministic training: Python random, NumPy,
PyTorch CPU/CUDA, cuDNN, and PYTHONHASHSEED.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, sets cuDNN to fully deterministic mode
            (slower but bit-exact reproducible). If False, uses
            benchmark mode for faster training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        logger.info("Seeded with %d (fully deterministic mode)", seed)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info("Seeded with %d (benchmark mode)", seed)
