"""
Dataset module for InfraNova AI.

Currently supports Landsat 9 TIR-RGB pairs.
"""

from typing import Optional
from torch.utils.data import DataLoader


def get_dataloader(
    dataset_name: str = "landsat9",
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 2,
    image_size: int = 256,
    shuffle: Optional[bool] = None,
    task: str = "colorization",
) -> DataLoader:
    """
    Create DataLoader for the specified dataset.
    """
    name = dataset_name.lower()
    
    if name == "landsat9":
        from .landsat9_dataset import Landsat9Dataset
        dataset = Landsat9Dataset(
            split=split,
            image_size=image_size,
            augment=(split == "train"),
            task=task,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if shuffle is None:
        shuffle = (split == "train")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=(split == "train"),
    )