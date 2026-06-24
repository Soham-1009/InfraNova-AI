"""
Landsat 9 Dataset for IR Super-Resolution and Colorization.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

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
    """
    
    def __init__(
        self,
        root_dir: str = "data/landsat9/splits",
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        task: str = "colorization",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.task = task
        
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        self.samples = sorted([
            d for d in self.split_dir.iterdir() if d.is_dir()
        ])
        
        if not self.samples:
            raise ValueError(f"No samples found in {self.split_dir}")
        
        logger.info(
            f"Loaded {len(self.samples)} samples for {split} split, task={task}"
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _normalize_tir(self, arr: np.ndarray) -> np.ndarray:
        """Normalize TIR to [-1, 1] using percentile stretching."""
        arr = arr.astype(np.float32)
        p_low, p_high = np.percentile(arr, (1, 99))
        if p_high - p_low < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip((arr - p_low) / (p_high - p_low), 0, 1)
        return arr * 2 - 1
    
    def _normalize_rgb(self, arr: np.ndarray) -> np.ndarray:
        """Normalize RGB to [-1, 1] using per-band percentile stretching."""
        normalized = []
        for band in arr:
            band = band.astype(np.float32)
            p_low, p_high = np.percentile(band, (1, 99))
            if p_high - p_low < 1e-8:
                normalized.append(np.zeros_like(band, dtype=np.float32))
                continue
            band = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            normalized.append(band * 2 - 1)
        return np.stack(normalized)
    
    def _resize(self, arr: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image to target size."""
        if arr.ndim == 2:
            return cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        else:
            bands = []
            for band in arr:
                bands.append(cv2.resize(band, (target_size, target_size), interpolation=cv2.INTER_CUBIC))
            return np.stack(bands)
    
    def _apply_augmentation(self, x: np.ndarray, y: np.ndarray):
        """Apply paired spatial augmentations."""
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=-1).copy()
            y = np.flip(y, axis=-1).copy()
        
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=-2).copy()
            y = np.flip(y, axis=-2).copy()
        
        k = np.random.randint(0, 4)
        if k > 0:
            x = np.rot90(x, k=k, axes=(-2, -1)).copy()
            y = np.rot90(y, k=k, axes=(-2, -1)).copy()
        
        return x, y
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_dir = self.samples[idx]
        
        tir_200m = np.load(sample_dir / 'tir_200m.npy')
        tir_100m = np.load(sample_dir / 'tir_100m.npy')
        rgb_100m = np.load(sample_dir / 'rgb_100m.npy')
        
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