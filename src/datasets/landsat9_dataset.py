"""
Landsat 9 Dataset for IR Super-Resolution and Colorization.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Landsat9Dataset(Dataset):
    """
    Landsat 9 dataset for IR-to-RGB colorization task.

    For colorization task:
    - Input: TIR 100m (128x128)
    - Output: RGB 100m (128x128)

    Each sample folder contains:
    - tir_200m.npy: (64, 64) thermal infrared at 200m
    - tir_100m.npy: (128, 128) thermal infrared at 100m
    - rgb_100m.npy: (3, 128, 128) RGB at 100m

    Supports two normalization modes:
    - "local" (default, backward-compatible): per-sample percentile stretching
    - "global": uses precomputed dataset-level statistics from a JSON file
    """

    def __init__(
        self,
        root_dir: str = "data/landsat9/splits",
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        task: str = "colorization",
        normalization: str = "local",
        stats_file: str | None = None,
        subset_ratio: float | None = None,
        subset_seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.task = task
        self.normalization = normalization

        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.samples = sorted([
            d for d in self.split_dir.iterdir() if d.is_dir()
        ])

        if not self.samples:
            raise ValueError(f"No samples found in {self.split_dir}")

        if subset_ratio is not None and 0.0 < subset_ratio < 1.0:
            import random
            rng = random.Random(subset_seed)
            # Create a deterministic copy and shuffle it
            shuffled_samples = list(self.samples)
            rng.shuffle(shuffled_samples)
            subset_size = max(1, int(len(shuffled_samples) * subset_ratio))
            self.samples = shuffled_samples[:subset_size]


        # Load global stats if requested
        self._global_stats: dict[str, Any] | None = None
        if normalization == "global":
            if stats_file is None:
                raise ValueError(
                    "normalization='global' requires stats_file path. "
                    "Run compute_normalization_stats.py first."
                )
            stats_path = Path(stats_file)
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"Global stats file not found: {stats_path}. "
                    f"Run: python compute_normalization_stats.py --output {stats_file}"
                )
            self._global_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            logger.info("Loaded global normalization stats from %s", stats_file)

        logger.info(
            "Loaded %d samples for %s split (subset_ratio=%s), task=%s, normalization=%s",
            len(self.samples), split, subset_ratio, task, normalization,
        )

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Normalization methods
    # ------------------------------------------------------------------

    def _normalize_tir_local(self, arr: np.ndarray) -> np.ndarray:
        """Normalize TIR to [-1, 1] using per-sample percentile stretching."""
        arr = arr.astype(np.float32)
        p_low, p_high = np.percentile(arr, (2, 98))
        if p_high - p_low < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip((arr - p_low) / (p_high - p_low), 0, 1)
        return arr * 2 - 1

    def _normalize_tir_global(self, arr: np.ndarray) -> np.ndarray:
        """Normalize TIR to [-1, 1] using global dataset statistics."""
        arr = arr.astype(np.float32)
        stats = self._global_stats["tir_100m"]
        p_low = stats["p2"]
        p_high = stats["p98"]
        if p_high - p_low < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip((arr - p_low) / (p_high - p_low), 0, 1)
        return arr * 2 - 1

    def _normalize_rgb_local(self, arr: np.ndarray) -> np.ndarray:
        """Normalize RGB to [-1, 1] using per-band, per-sample percentile stretching."""
        normalized = []
        for band in arr:
            band = band.astype(np.float32)
            p_low, p_high = np.percentile(band, (2, 98))
            if p_high - p_low < 1e-8:
                normalized.append(np.zeros_like(band, dtype=np.float32))
                continue
            band = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            normalized.append(band * 2 - 1)
        return np.stack(normalized)

    def _normalize_rgb_global(self, arr: np.ndarray) -> np.ndarray:
        """Normalize RGB to [-1, 1] using global dataset statistics per band."""
        rgb_stats = self._global_stats["rgb_100m"]
        band_names = ["red", "green", "blue"]
        normalized = []
        for band_idx, name in enumerate(band_names):
            band = arr[band_idx].astype(np.float32)
            s = rgb_stats[name]
            p_low = s["p2"]
            p_high = s["p98"]
            if p_high - p_low < 1e-8:
                normalized.append(np.zeros_like(band, dtype=np.float32))
                continue
            band = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            normalized.append(band * 2 - 1)
        return np.stack(normalized)

    def _normalize_tir(self, arr: np.ndarray) -> np.ndarray:
        """Dispatch to the selected normalization mode for TIR."""
        if self.normalization == "global" and self._global_stats is not None:
            return self._normalize_tir_global(arr)
        return self._normalize_tir_local(arr)

    def _normalize_rgb(self, arr: np.ndarray) -> np.ndarray:
        """Dispatch to the selected normalization mode for RGB."""
        if self.normalization == "global" and self._global_stats is not None:
            return self._normalize_rgb_global(arr)
        return self._normalize_rgb_local(arr)

    # ------------------------------------------------------------------
    # Spatial operations
    # ------------------------------------------------------------------

    def _resize(self, arr: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image to target size."""
        if arr.ndim == 2:
            return cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        else:
            bands = []
            for band in arr:
                bands.append(cv2.resize(band, (target_size, target_size), interpolation=cv2.INTER_CUBIC))
            return np.stack(bands)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _apply_augmentation(self, x: np.ndarray, y: np.ndarray):
        """
        Apply paired spatial and radiometric augmentations.

        Spatial augmentations (applied identically to input and target):
        - Horizontal flip (p=0.5)
        - Vertical flip (p=0.5)
        - Random 90° rotation (uniform k in {0,1,2,3})

        Radiometric augmentations (applied identically to preserve pairing):
        - Random brightness shift ±10% (p=0.3)
        - Random contrast adjustment ±10% (p=0.3)

        Input-only augmentations:
        - Gaussian noise on TIR input (p=0.3, sigma=0.02)
        """
        # --- Spatial: horizontal flip ---
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=-1).copy()
            y = np.flip(y, axis=-1).copy()

        # --- Spatial: vertical flip ---
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=-2).copy()
            y = np.flip(y, axis=-2).copy()

        # --- Spatial: random 90° rotation ---
        k = np.random.randint(0, 4)
        if k > 0:
            x = np.rot90(x, k=k, axes=(-2, -1)).copy()
            y = np.rot90(y, k=k, axes=(-2, -1)).copy()

        # --- Radiometric: paired brightness shift ---
        if np.random.rand() < 0.3:
            factor = 1.0 + np.random.uniform(-0.1, 0.1)
            x = x * factor
            y = y * factor

        # --- Radiometric: paired contrast adjustment ---
        if np.random.rand() < 0.3:
            factor = 1.0 + np.random.uniform(-0.1, 0.1)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            x = (x - x_mean) * factor + x_mean
            y = (y - y_mean) * factor + y_mean

        # --- Input-only: Gaussian noise on TIR (simulate sensor noise) ---
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.02, size=x.shape).astype(x.dtype)
            x = x + noise

        return x, y

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_dir = self.samples[idx]

        tir_200m = np.load(sample_dir / 'tir_200m.npy')
        tir_100m = np.load(sample_dir / 'tir_100m.npy')
        rgb_100m = np.load(sample_dir / 'rgb_100m.npy')

        # STRICT SHAPE VALIDATION (No silent resizing)
        if tir_100m.shape[-2:] != (self.image_size, self.image_size):
            raise RuntimeError(
                f"TIR spatial shape {tir_100m.shape[-2:]} does not match configured image_size ({self.image_size}, {self.image_size})."
            )
        if rgb_100m.shape[-2:] != (self.image_size, self.image_size):
            raise RuntimeError(
                f"RGB spatial shape {rgb_100m.shape[-2:]} does not match configured image_size ({self.image_size}, {self.image_size})."
            )

        # Ensure RGB channel count matches 3, agnostic to CHW vs HWC
        if rgb_100m.shape[0] != 3 and rgb_100m.shape[-1] != 3:
            raise RuntimeError(f"RGB array does not appear to contain 3 channels: {rgb_100m.shape}")

        # Ensure correct data type and no NaNs
        if rgb_100m.dtype != np.float32 or tir_100m.dtype != np.float32:
            raise RuntimeError("Arrays must be float32.")
        if not np.isfinite(rgb_100m).all() or not np.isfinite(tir_100m).all():
            raise RuntimeError("Arrays must not contain NaNs or Infs.")

        if self.task == "colorization":
            input_arr = tir_100m
            target_arr = rgb_100m
        elif self.task == "super_resolution":
            input_arr = tir_200m
            target_arr = tir_100m[np.newaxis, ...]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if self.augment:
            if input_arr.ndim == 2:
                input_arr_aug = input_arr[np.newaxis, ...]
            else:
                input_arr_aug = input_arr

            input_arr_aug, target_arr = self._apply_augmentation(input_arr_aug, target_arr)

            if input_arr.ndim == 2:
                input_arr = input_arr_aug[0]
            else:
                input_arr = input_arr_aug

        input_arr = self._resize(input_arr, self.image_size)
        target_arr = self._resize(target_arr, self.image_size)

        if input_arr.ndim == 2:
            input_arr = self._normalize_tir(input_arr)
            input_tensor = torch.from_numpy(input_arr).unsqueeze(0).float()
        else:
            input_arr = self._normalize_rgb(input_arr) if input_arr.shape[0] == 3 else self._normalize_tir(input_arr)
            input_tensor = torch.from_numpy(input_arr).float()

        if target_arr.ndim == 2:
            target_arr = self._normalize_tir(target_arr)
            target_tensor = torch.from_numpy(target_arr).unsqueeze(0).float()
        else:
            if target_arr.shape[0] == 3:
                target_arr = self._normalize_rgb(target_arr)
            else:
                target_arr = self._normalize_tir(target_arr[0])[np.newaxis, ...]
            target_tensor = torch.from_numpy(target_arr).float()

        return {
            'ir': input_tensor,
            'rgb': target_tensor,
            'name': sample_dir.name,
        }
