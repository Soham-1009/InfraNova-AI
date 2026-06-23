from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader

from .sentinel2_dataset import Sentinel2Dataset
from .sentinel2_multispectral_dataset import Sentinel2MultispectralDataset
from .transforms import get_train_transforms, get_val_transforms


def get_dataloader(
    dataset_name: str = "sentinel2",
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 256,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Returns optimized DataLoader for the chosen dataset.

    Supported datasets:
        - sentinel2
        - multispectral

    Args:
        dataset_name: dataset identifier
        split: train / val / test
        batch_size: batch size
        num_workers: dataloader workers
        image_size: target image size
        shuffle: optional override

    Returns:
        torch DataLoader
    """
    name = dataset_name.lower()

    if name == "sentinel2":
        transforms = get_train_transforms(image_size) if split == "train" else get_val_transforms(image_size)
        dataset = Sentinel2Dataset(
            split=split,
            image_size=image_size,
            transform=transforms,
            paired=True,
        )

    elif name == "multispectral":
        dataset = Sentinel2MultispectralDataset(
            split=split,
            image_size=image_size,
            augment=(split == "train"),
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=(split == "train"),
    )