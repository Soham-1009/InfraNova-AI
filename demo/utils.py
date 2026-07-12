from __future__ import annotations

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.image_processing import to_single_band_array


def _normalize_to_uint8(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Percentile-stretch a single-channel array to uint8 for display."""
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


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
    target_size = int(image_size)
    if target_size <= 0:
        raise ValueError("image_size must be a positive integer")

    arr = to_single_band_array(image)

    lo = np.percentile(arr, 2.0)
    hi = np.percentile(arr, 98.0)
    if hi - lo < 1e-6:
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr.astype(np.float32) * 2.0 - 1.0

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
        if tensor.size(0) != 1:
            raise ValueError("postprocess_output expects a batch containing exactly one image")
        tensor = tensor[0]
    if tensor.dim() != 3 or tensor.size(0) != 3:
        raise ValueError(
            "postprocess_output expects a tensor shaped [3, H, W] or [1, 3, H, W]"
        )

    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255.0).round().astype(np.uint8)

    return Image.fromarray(tensor)


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

    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge([l_channel, a, b])
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


def visualize_tir_as_thermal(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """
    Convert grayscale TIR to thermal colormap for better visualization.

    Args:
        image: PIL grayscale or numpy array.

    Returns:
        PIL RGB image with thermal colormap applied.
    """
    arr = _normalize_to_uint8(to_single_band_array(image))

    # Apply thermal colormap (INFERNO gives a good heat-like appearance)
    colored = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)

    # Convert BGR to RGB
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return Image.fromarray(colored_rgb)


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
                with Image.open(path) as image:
                    image.load()
                    images.append(image.copy())
            except Exception:
                continue

    return images
