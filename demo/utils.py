from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.image_processing import (
    DEFAULT_PERCENTILE_HIGH,
    DEFAULT_PERCENTILE_LOW,
    to_single_band_array,
)


def _normalize_to_uint8(
    arr: np.ndarray,
    low: float = DEFAULT_PERCENTILE_LOW,
    high: float = DEFAULT_PERCENTILE_HIGH,
) -> np.ndarray:
    """Percentile-stretch a single-channel array to uint8 for display."""
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def preprocess_ir_image(
    image: Image.Image | np.ndarray | tuple[np.ndarray, np.ndarray],
    image_size: int = 128,
    target_channels: int = 1,
) -> torch.Tensor:
    """
    Preprocess IR image(s) for Pix2Pix inference.

    Supports:
        - 2-channel Landsat 9 thermal arrays (shape [2, H, W] or [H, W, 2])
        - Tuple / list of (band10, band11) arrays
        - Single-channel PIL image or array (duplicated if target_channels=2)

    Returns:
        Tensor of shape [1, target_channels, image_size, image_size] in [-1, 1].
    """
    target_size = int(image_size)
    if target_size <= 0:
        raise ValueError("image_size must be a positive integer")

    # Case 1: Tuple/List of (Band 10, Band 11)
    if isinstance(image, (tuple, list)) and len(image) == 2:
        b10, b11 = image
        b10_arr = to_single_band_array(b10)
        b11_arr = to_single_band_array(b11)
        bands = [b10_arr, b11_arr]
    elif isinstance(image, np.ndarray) and image.ndim == 3:
        # Channel-first [2, H, W]
        if image.shape[0] == 2:
            bands = [image[0], image[1]]
        elif image.shape[2] == 2:
            bands = [image[:, :, 0], image[:, :, 1]]
        elif image.shape[0] >= 3:
            arr = to_single_band_array(image)
            bands = [arr, arr] if target_channels == 2 else [arr]
        else:
            arr = to_single_band_array(image)
            bands = [arr, arr] if target_channels == 2 else [arr]
    else:
        # Single band input
        arr = to_single_band_array(image)
        bands = [arr, arr] if target_channels == 2 else [arr]

    processed_bands = []
    for b in bands[:target_channels]:
        b_f = np.asarray(b, dtype=np.float32)
        lo = np.percentile(b_f, DEFAULT_PERCENTILE_LOW)
        hi = np.percentile(b_f, DEFAULT_PERCENTILE_HIGH)
        if hi - lo < 1e-6:
            norm_b = np.zeros_like(b_f, dtype=np.float32)
        else:
            norm_b = np.clip((b_f - lo) / (hi - lo), 0.0, 1.0)
        norm_b = cv2.resize(norm_b, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        norm_b = np.clip(norm_b, 0.0, 1.0)
        norm_b = norm_b.astype(np.float32) * 2.0 - 1.0
        processed_bands.append(norm_b)

    # Stack to [C, H, W]
    stacked = np.stack(processed_bands, axis=0)  # [C, H, W]
    tensor = torch.from_numpy(stacked).unsqueeze(0)  # [1, C, H, W]
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
        String path to saved image.
    """
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return str(out_path)


def visualize_tir_as_thermal(image: Image.Image | np.ndarray) -> Image.Image:
    """
    Convert a single-channel TIR array to a colorized thermal visualization.

    Applies percentile stretch + OpenCV INFERNO colormap.

    Args:
        image: Input TIR image (PIL or numpy).

    Returns:
        PIL RGB image with INFERNO colormap applied.
    """
    arr = to_single_band_array(image)
    norm = _normalize_to_uint8(arr)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_rgb)
