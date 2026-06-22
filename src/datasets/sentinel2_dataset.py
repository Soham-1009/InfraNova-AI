"""
Sentinel-2 IR-RGB Dataset for Pix2Pix Training.

Author: Soham Deshpande
Project: ISRO BAH 2026 - InfraNova AI
"""

import logging
import random
from pathlib import Path
from typing import Dict, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Sentinel2Dataset(Dataset):
    """
    Sentinel-2 IR -> RGB Dataset for Pix2Pix Training.
    
    Args:
        root_dir: Path to dataset root
        split: 'train', 'val', or 'test'
        image_size: Target size (resized from 64x64)
        transform: Albumentations Compose object
        paired: True for Pix2Pix, False for CycleGAN
    
    Returns:
        dict: {
            'ir':   Tensor [1, H, W] in [-1, 1] float32,
            'rgb':  Tensor [3, H, W] in [-1, 1] float32,
            'name': str
        }
    """
    
    def __init__(
        self,
        root_dir: str = "data/raw/sentinel2",
        split: str = "train",
        image_size: int = 256,
        transform: Optional[A.Compose] = None,
        paired: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.paired = paired
        
        self.ir_dir = self.root_dir / split / "ir"
        self.rgb_dir = self.root_dir / split / "rgb"
        
        if not self.ir_dir.exists():
            raise FileNotFoundError(f"IR directory not found: {self.ir_dir}")
        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        
        self.ir_files = sorted(list(self.ir_dir.glob("*.png")))
        self.rgb_files = sorted(list(self.rgb_dir.glob("*.png")))
        
        if paired:
            rgb_names = {p.name for p in self.rgb_files}
            self.samples = []
            for ir_file in self.ir_files:
                if ir_file.name in rgb_names:
                    self.samples.append({
                        "ir": ir_file,
                        "rgb": self.rgb_dir / ir_file.name,
                    })
            
            logger.info(
                "Loaded %d paired samples for %s split",
                len(self.samples),
                split,
            )
            
            if len(self.samples) == 0:
                raise ValueError(
                    f"No paired samples found! "
                    f"Check {self.ir_dir} and {self.rgb_dir}"
                )
        else:
            logger.info(
                "Loaded %d IR and %d RGB images in unpaired mode",
                len(self.ir_files),
                len(self.rgb_files),
            )
    
    def __len__(self) -> int:
        if self.paired:
            return len(self.samples)
        return max(len(self.ir_files), len(self.rgb_files))
    
    def _load_ir(self, path: Path) -> np.ndarray:
        """Load IR image as grayscale numpy array."""
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load IR image: {path}")
        return image
    
    def _load_rgb(self, path: Path) -> np.ndarray:
        """Load RGB image, convert BGR -> RGB."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load RGB image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _normalize_to_range(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize uint8 tensor [0, 255] to float32 [-1, 1]."""
        return tensor.float() / 127.5 - 1.0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns dict with keys 'ir', 'rgb', 'name'."""
        max_retries = 5
        retries = 0
        
        while retries < max_retries:
            try:
                if self.paired:
                    sample = self.samples[idx]
                    ir_path = sample["ir"]
                    rgb_path = sample["rgb"]
                else:
                    ir_path = self.ir_files[idx % len(self.ir_files)]
                    rgb_path = random.choice(self.rgb_files)
                
                ir_image = self._load_ir(ir_path)
                rgb_image = self._load_rgb(rgb_path)
                
                if self.transform is not None:
                    transformed = self.transform(
                        image=ir_image,
                        rgb=rgb_image,
                    )
                    ir_tensor = transformed["image"]
                    rgb_tensor = transformed["rgb"]
                    
                    if ir_tensor.dim() == 2:
                        ir_tensor = ir_tensor.unsqueeze(0)
                    
                    # Normalize after augmentation
                    ir_tensor = self._normalize_to_range(ir_tensor)
                    rgb_tensor = self._normalize_to_range(rgb_tensor)
                else:
                    # No transform - manual conversion
                    ir_tensor = self._normalize_to_range(
                        torch.from_numpy(ir_image)
                    ).unsqueeze(0)
                    rgb_tensor = self._normalize_to_range(
                        torch.from_numpy(rgb_image)
                    ).permute(2, 0, 1)
                
                return {
                    "ir": ir_tensor,
                    "rgb": rgb_tensor,
                    "name": ir_path.name,
                }
            
            except Exception as exc:
                logger.warning(
                    "Skipping corrupted sample at index %d: %s",
                    idx,
                    str(exc),
                )
                idx = random.randint(0, len(self) - 1)
                retries += 1
        
        raise RuntimeError(
            f"Failed to load sample after {max_retries} retries"
        )