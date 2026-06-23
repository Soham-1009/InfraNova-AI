from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_BANDS = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12]
TARGET_BANDS = [3, 2, 1]


class Sentinel2MultispectralDataset(Dataset):
    """
    EuroSAT multispectral dataset for IR-to-RGB translation.

    Loads .npy files of shape (13, 64, 64) uint16 and returns:
        {
            "input": Tensor [10, 256, 256],
            "target": Tensor [3, 256, 256],
            "name": str
        }

    Input bands:
        [0, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    Target bands:
        [3, 2, 1]  -> RGB
    """

    def __init__(
        self,
        root_dir: str = "data/multispectral",
        split: str = "train",
        image_size: int = 256,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        augment: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = int(image_size)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.augment = bool(augment)

        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.files = sorted(self.split_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _resize_band(band: np.ndarray, image_size: int) -> np.ndarray:
        return cv2.resize(
            band,
            (image_size, image_size),
            interpolation=cv2.INTER_CUBIC,
        )

    def _percentile_normalize(self, band: np.ndarray) -> np.ndarray:
        """
        Normalize a single band to [-1, 1] using per-band percentiles.
        """
        band = band.astype(np.float32)

        lo = np.percentile(band, self.percentile_low)
        hi = np.percentile(band, self.percentile_high)

        if hi - lo < 1e-6:
            return np.zeros_like(band, dtype=np.float32)

        band = np.clip(band, lo, hi)
        band = (band - lo) / (hi - lo)
        band = band * 2.0 - 1.0
        return band.astype(np.float32)

    @staticmethod
    def _apply_pair_augmentation(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply identical spatial augmentations to input and target.
        """
        # Horizontal flip
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=2).copy()

        # Vertical flip
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k:
            x = np.rot90(x, k=k, axes=(1, 2)).copy()
            y = np.rot90(y, k=k, axes=(1, 2)).copy()

        return x, y

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        arr = np.load(path)

        if arr.shape != (13, 64, 64):
            raise ValueError(f"Invalid multispectral shape in {path}: {arr.shape}")

        x = arr[INPUT_BANDS]
        y = arr[TARGET_BANDS]

        if self.augment and self.split == "train":
            x, y = self._apply_pair_augmentation(x, y)

        x_out = []
        y_out = []

        for band in x:
            band = self._resize_band(band, self.image_size)
            band = self._percentile_normalize(band)
            x_out.append(band)

        for band in y:
            band = self._resize_band(band, self.image_size)
            band = self._percentile_normalize(band)
            y_out.append(band)

        x_tensor = torch.from_numpy(np.stack(x_out, axis=0)).float()
        y_tensor = torch.from_numpy(np.stack(y_out, axis=0)).float()

        return {
            "input": x_tensor,
            "target": y_tensor,
            "name": path.stem,
        }