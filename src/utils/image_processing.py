from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image


ImageInput = Union[Image.Image, np.ndarray]


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
