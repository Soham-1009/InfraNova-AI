from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

import cv2
import numpy as np
import torch
from PIL import Image


def preprocess_ir_image(
    image: Union[Image.Image, np.ndarray],
    image_size: int = 256,
) -> torch.Tensor:
    """
    Preprocess an IR image for Pix2Pix inference.

    Args:
        image: PIL image or numpy array.
        image_size: Target size (square resize).

    Returns:
        Tensor of shape [1, 1, image_size, image_size] in [-1, 1].
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("L"))
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise TypeError("image must be PIL.Image.Image or numpy.ndarray")

    arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # [-1, 1]

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return tensor


def postprocess_output(tensor: torch.Tensor) -> Image.Image:
    """
    Convert model output tensor to displayable PIL RGB image.

    Args:
        tensor: Tensor [3, H, W] or [1, 3, H, W] in [-1, 1].

    Returns:
        PIL RGB image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255.0).round().astype(np.uint8)

    return Image.fromarray(tensor, mode="RGB")


def enhance_output(image: Image.Image) -> Image.Image:
    """
    Improve contrast of generated RGB output using CLAHE on luminance channel.

    Args:
        image: PIL RGB image.

    Returns:
        Enhanced PIL RGB image.
    """
    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)


def save_output(image: Image.Image, filename: str) -> str:
    """
    Save image to disk, creating parent directories as needed.

    Args:
        image: PIL image to save.
        filename: Target path.

    Returns:
        Saved file path.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def load_sample_images(sample_dir: str = "demo/assets/samples") -> List[Image.Image]:
    """
    Load sample IR images for demo/testing.

    Args:
        sample_dir: Directory containing sample images.

    Returns:
        List of PIL grayscale images.
    """
    directory = Path(sample_dir)
    if not directory.exists():
        return []

    images: List[Image.Image] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        for path in sorted(directory.glob(ext)):
            try:
                images.append(Image.open(path).convert("L"))
            except Exception:
                continue

    return images