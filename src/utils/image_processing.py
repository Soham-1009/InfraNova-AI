from __future__ import annotations

from typing import Union

import numpy as np
import torch
from PIL import Image


ImageInput = Union[Image.Image, np.ndarray]

DEFAULT_PERCENTILE_LOW = 1.0
DEFAULT_PERCENTILE_HIGH = 99.0


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW/BCHW tensor to HWC/BHWC NumPy layout on the CPU."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")

    array = tensor.detach().cpu().numpy()
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    if array.ndim == 3:
        return np.ascontiguousarray(np.moveaxis(array, 0, -1))
    if array.ndim == 4:
        return np.ascontiguousarray(np.moveaxis(array, 1, -1))
    raise ValueError(
        "Expected a 2D, 3D, or 4D tensor, "
        f"received shape {tuple(tensor.shape)}."
    )


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert an HW/HWC/BHWC NumPy image array to CHW/BCHW float32 tensor layout."""
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("array must have a numeric dtype")

    if array.ndim == 2:
        channels_first = array[np.newaxis, ...]
    elif array.ndim == 3:
        first_is_channel = array.shape[0] in (1, 3, 4)
        last_is_channel = array.shape[-1] in (1, 3, 4)
        if last_is_channel and not first_is_channel:
            channels_first = np.moveaxis(array, -1, 0)
        elif first_is_channel and not last_is_channel:
            channels_first = array
        elif last_is_channel:
            channels_first = np.moveaxis(array, -1, 0)
        else:
            raise ValueError(
                "Expected a channel-first or channel-last image with 1, 3, or 4 channels; "
                f"received shape {array.shape}."
            )
    elif array.ndim == 4:
        second_is_channel = array.shape[1] in (1, 3, 4)
        last_is_channel = array.shape[-1] in (1, 3, 4)
        if last_is_channel and not second_is_channel:
            channels_first = np.moveaxis(array, -1, 1)
        elif second_is_channel:
            channels_first = array
        else:
            raise ValueError(
                "Expected a BCHW or BHWC batch with 1, 3, or 4 channels; "
                f"received shape {array.shape}."
            )
    else:
        raise ValueError(f"Expected a 2D, 3D, or 4D array, received shape {array.shape}.")

    return torch.from_numpy(np.ascontiguousarray(channels_first, dtype=np.float32))


def to_single_band_array(image: ImageInput) -> np.ndarray:
    """Return an image as a finite 2D float32 array without reducing its precision."""
    if isinstance(image, Image.Image):
        # Palette images contain palette indices rather than meaningful band values.
        source = image.convert("RGB") if image.mode == "P" else image
        arr = np.asarray(source)
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError("image must be a PIL.Image.Image or numpy.ndarray")

    if arr.ndim == 2:
        single_band = arr
    elif arr.ndim == 3:
        first, last = arr.shape[0], arr.shape[-1]
        first_is_ch = first in (1, 3, 4)
        last_is_ch = last in (1, 3, 4)

        if first_is_ch and not last_is_ch:
            # Unambiguous channel-first: e.g. (3, 128, 128)
            channels = np.moveaxis(arr, 0, -1)
        elif first_is_ch and last_is_ch and first != last:
            # Ambiguous but shape[0] looks like a channel count while
            # shape[-1] looks like a small spatial dim: e.g. (3, 1, 1)
            channels = np.moveaxis(arr, 0, -1)
        elif last_is_ch:
            # Channel-last: e.g. (128, 128, 3)
            channels = arr
        else:
            raise ValueError(
                "Expected a grayscale image or an image with 1, 3, or 4 channels; "
                f"received shape {arr.shape}."
            )

        if channels.shape[-1] == 1:
            single_band = channels[..., 0]
        else:
            rgb = channels[..., :3].astype(np.float32, copy=False)
            single_band = (
                0.299 * rgb[..., 0]
                + 0.587 * rgb[..., 1]
                + 0.114 * rgb[..., 2]
            )
    else:
        raise ValueError(f"Expected a 2D or 3D image array, received shape {arr.shape}.")

    single_band = np.asarray(single_band, dtype=np.float32)
    finite = np.isfinite(single_band)
    if not finite.any():
        raise ValueError("Input image contains no finite pixel values.")
    if not finite.all():
        replacement = float(np.median(single_band[finite]))
        single_band = np.where(finite, single_band, replacement)

    return np.ascontiguousarray(single_band)
