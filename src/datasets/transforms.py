"""
Data Augmentation Transforms for Sentinel-2 IR-RGB Dataset.

Mathematically validated augmentations (per DeepSeek analysis):
- Only geometric transforms that preserve f: IR -> RGB mapping
- NO ColorJitter (proven to break deterministic mapping)
- Manual normalization in dataset class (more reliable than custom transform)

Author: Soham Deshpande
Project: ISRO BAH 2026 - InfraNova AI
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 256) -> A.Compose:
    """
    Training transforms with paired augmentations.
    
    Spatial augmentations only (preserve IR -> RGB mapping):
    - HorizontalFlip
    - VerticalFlip
    - RandomRotate90
    
    Normalization is handled manually in dataset __getitem__
    to ensure proper handling of 1-channel IR and 3-channel RGB.
    """
    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=3,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ],
        additional_targets={
            "rgb": "image",
        },
    )


def get_val_transforms(image_size: int = 256) -> A.Compose:
    """
    Validation/Test transforms - no augmentation, just resize.
    Normalization handled in dataset __getitem__.
    """
    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=3,
            ),
            ToTensorV2(),
        ],
        additional_targets={
            "rgb": "image",
        },
    )