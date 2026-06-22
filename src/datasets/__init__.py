"""
Datasets module for InfraNova AI.

Provides factory function to create optimized DataLoaders for training.

Usage:
    from src.datasets import get_dataloader
    
    train_loader = get_dataloader(
        dataset_name='sentinel2',
        split='train',
        batch_size=8,
    )
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader

from .sentinel2_dataset import Sentinel2Dataset
from .transforms import (
    get_train_transforms,
    get_val_transforms,
)


def get_dataloader(
    dataset_name: str = "sentinel2",
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 2,
    image_size: int = 256,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Creates optimized DataLoader for Google Colab T4 GPU.
    
    Args:
        dataset_name: Dataset identifier (currently only 'sentinel2')
        split: 'train', 'val', or 'test'
        batch_size: Batch size (default: 8, optimized for T4 + AMP)
        num_workers: Number of dataloader workers (default: 2)
        image_size: Target image size (default: 256)
        shuffle: Optional override for shuffling.
                 If None, auto-shuffles for train, no shuffle for val/test.
    
    Returns:
        torch.utils.data.DataLoader: Configured DataLoader
    
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() != "sentinel2":
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Currently only 'sentinel2' is supported."
        )
    
    # Use train transforms (with augmentation) or val transforms (deterministic)
    if split == "train":
        transforms = get_train_transforms(image_size)
    else:
        transforms = get_val_transforms(image_size)
    
    # Create dataset
    dataset = Sentinel2Dataset(
        split=split,
        image_size=image_size,
        transform=transforms,
        paired=True,
    )
    
    # Auto-determine shuffle behavior
    if shuffle is None:
        shuffle = (split == "train")
    
    # Return optimized DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=(split == "train"),
    )


# Export public API
__all__ = [
    "Sentinel2Dataset",
    "get_train_transforms",
    "get_val_transforms",
    "get_dataloader",
]