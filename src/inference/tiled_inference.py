"""
Tiled whole-raster inference and smooth blending for Landsat 9 Pix2PixHD.

Enables seamless colorization of large, arbitrary-dimension satellite scenes
(e.g., 1000x1000 or full scenes) without tile boundary artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import tifffile
except ImportError:
    tifffile = None

from demo.inference import InferenceEngine
from src.utils.image_processing import (
    DEFAULT_PERCENTILE_HIGH,
    DEFAULT_PERCENTILE_LOW,
    to_single_band_array,
)


def create_2d_window(tile_size: int, window_type: str = "cosine") -> np.ndarray:
    """
    Generate a 2D weighting window for smooth patch stitching.
    """
    if window_type == "cosine":
        # 1D Hann/Cosine window
        w1d = np.sin(np.linspace(0, np.pi, tile_size)) ** 2
        w2d = np.outer(w1d, w1d)
    elif window_type == "linear":
        # 1D Triangular / Bartlett window
        half = tile_size // 2
        w1d = np.concatenate([np.linspace(0.1, 1.0, half), np.linspace(1.0, 0.1, tile_size - half)])
        w2d = np.outer(w1d, w1d)
    else:
        w2d = np.ones((tile_size, tile_size), dtype=np.float32)

    # Avoid zero division on extreme edges
    w2d = np.maximum(w2d, 1e-4).astype(np.float32)
    return w2d


class TiledRasterInference:
    """
    Inference manager for large, arbitrary-size whole-raster Landsat 9 imagery.
    """

    def __init__(
        self,
        checkpoint_path: str = "outputs/best/pix2pix_landsat_best.pth",
        tile_size: int = 128,
        overlap: int = 32,
        batch_size: int = 8,
        device: str | torch.device | None = None,
        window_type: str = "cosine",
        percentile_low: float = DEFAULT_PERCENTILE_LOW,
        percentile_high: float = DEFAULT_PERCENTILE_HIGH,
    ) -> None:
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.stride = self.tile_size - self.overlap
        if self.stride <= 0:
            raise ValueError(f"Overlap ({overlap}) must be strictly less than tile_size ({tile_size})")

        self.batch_size = int(batch_size)
        self.window_type = window_type
        self.window_weights = create_2d_window(self.tile_size, window_type=window_type)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)

        self.engine = InferenceEngine(
            checkpoint_path=checkpoint_path,
            image_size=self.tile_size,
            device=device,
        )
        self.engine.load_model()
        self.device = self.engine.device

    @staticmethod
    def load_raster(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Load raster file (TIFF / GeoTIFF / NPY / PNG).
        Returns array [H, W] or [C, H, W] and georeferencing metadata.
        """
        path = Path(path)
        meta: dict[str, Any] = {}

        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            return arr, meta

        if rasterio is not None:
            with rasterio.open(path) as src:
                arr = src.read()
                meta = src.meta.copy()
                if arr.shape[0] == 1:
                    arr = arr[0]
                return arr, meta
        elif tifffile is not None and path.suffix.lower() in (".tif", ".tiff"):
            arr = tifffile.imread(path)
            return arr, meta
        else:
            pil_img = Image.open(path)
            arr = np.array(pil_img)
            return arr, meta

    def preprocess_scene(
        self,
        band10: np.ndarray,
        band11: np.ndarray,
    ) -> np.ndarray:
        """
        Apply global scene-level percentile stretching and normalization to [-1, 1].
        Returns normalized 2-band float32 array of shape [2, H, W].
        """
        b10 = band10.astype(np.float32)
        b11 = band11.astype(np.float32)

        # Normalize Band 10
        lo10 = np.percentile(b10, self.percentile_low)
        hi10 = np.percentile(b10, self.percentile_high)
        if hi10 - lo10 < 1e-6:
            norm10 = np.zeros_like(b10, dtype=np.float32)
        else:
            norm10 = np.clip((b10 - lo10) / (hi10 - lo10), 0.0, 1.0)
        norm10 = norm10 * 2.0 - 1.0

        # Normalize Band 11
        lo11 = np.percentile(b11, self.percentile_low)
        hi11 = np.percentile(b11, self.percentile_high)
        if hi11 - lo11 < 1e-6:
            norm11 = np.zeros_like(b11, dtype=np.float32)
        else:
            norm11 = np.clip((b11 - lo11) / (hi11 - lo11), 0.0, 1.0)
        norm11 = norm11 * 2.0 - 1.0

        return np.stack([norm10, norm11], axis=0)  # [2, H, W]

    @torch.inference_mode()
    def predict_scene(
        self,
        band10_input: str | Path | np.ndarray,
        band11_input: str | Path | np.ndarray,
        use_tta: bool = False,
    ) -> dict[str, Any]:
        """
        Perform tiled, overlap-blended inference on an entire satellite scene.

        Returns:
            {
                "rgb_uint8": np.ndarray [H, W, 3] (0..255),
                "rgb_float": np.ndarray [H, W, 3] (-1..1),
                "weight_map": np.ndarray [H, W],
                "tiles_processed": int,
                "scene_shape": tuple[int, int],
                "meta": dict,
            }
        """
        meta = {}
        if isinstance(band10_input, (str, Path)):
            b10_arr, meta = self.load_raster(band10_input)
        else:
            b10_arr = band10_input

        if isinstance(band11_input, (str, Path)):
            b11_arr, _ = self.load_raster(band11_input)
        else:
            b11_arr = band11_input

        # Ensure 2D arrays [H, W]
        b10_arr = to_single_band_array(b10_arr)
        b11_arr = to_single_band_array(b11_arr)

        if b10_arr.shape != b11_arr.shape:
            # Resize b11 to match b10 if there is minor resolution mismatch
            b11_arr = cv2.resize(b11_arr, (b10_arr.shape[1], b10_arr.shape[0]), interpolation=cv2.INTER_CUBIC)

        H, W = b10_arr.shape
        scene_normalized = self.preprocess_scene(b10_arr, b11_arr)  # [2, H, W]

        # Calculate grid coordinates
        y_starts = list(range(0, max(1, H - self.tile_size + 1), self.stride))
        if y_starts[-1] + self.tile_size < H:
            y_starts.append(H - self.tile_size)

        x_starts = list(range(0, max(1, W - self.tile_size + 1), self.stride))
        if x_starts[-1] + self.tile_size < W:
            x_starts.append(W - self.tile_size)

        # Preallocate accumulation buffers
        rgb_accum = np.zeros((3, H, W), dtype=np.float32)
        weight_accum = np.zeros((H, W), dtype=np.float32)

        # Collect tiles and coordinates
        tile_coords: list[tuple[int, int]] = []
        tile_tensors: list[torch.Tensor] = []

        for y in y_starts:
            for x in x_starts:
                patch = scene_normalized[:, y : y + self.tile_size, x : x + self.tile_size]
                # If scene is smaller than tile_size, pad
                if patch.shape[1] < self.tile_size or patch.shape[2] < self.tile_size:
                    pad_h = self.tile_size - patch.shape[1]
                    pad_w = self.tile_size - patch.shape[2]
                    patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

                tile_coords.append((y, x))
                tile_tensors.append(torch.from_numpy(patch))

        # Process in batches
        num_tiles = len(tile_tensors)
        model = self.engine.model

        for i in range(0, num_tiles, self.batch_size):
            batch_coords = tile_coords[i : i + self.batch_size]
            batch_tensors = torch.stack(tile_tensors[i : i + self.batch_size]).to(self.device)

            if not use_tta:
                out = model.generate(batch_tensors)
            else:
                preds = [model.generate(batch_tensors)]
                # Flip H
                out_h = model.generate(torch.flip(batch_tensors, dims=[3]))
                preds.append(torch.flip(out_h, dims=[3]))
                # Flip V
                out_v = model.generate(torch.flip(batch_tensors, dims=[2]))
                preds.append(torch.flip(out_v, dims=[2]))
                out = torch.stack(preds).mean(dim=0)

            out_np = out.detach().cpu().numpy()  # [B, 3, 128, 128]

            for b, (y, x) in enumerate(batch_coords):
                pred_tile = out_np[b]  # [3, 128, 128]
                valid_h = min(self.tile_size, H - y)
                valid_w = min(self.tile_size, W - x)

                w_slice = self.window_weights[:valid_h, :valid_w]
                for c in range(3):
                    rgb_accum[c, y : y + valid_h, x : x + valid_w] += pred_tile[c, :valid_h, :valid_w] * w_slice
                weight_accum[y : y + valid_h, x : x + valid_w] += w_slice

        # Normalize by accumulated weights
        weight_safe = np.maximum(weight_accum, 1e-6)
        for c in range(3):
            rgb_accum[c] /= weight_safe

        # Transpose to [H, W, 3] in [-1, 1]
        rgb_float = np.transpose(rgb_accum, (1, 2, 0))
        rgb_float = np.clip(rgb_float, -1.0, 1.0)

        # Convert to uint8 [0, 255]
        rgb_uint8 = ((rgb_float + 1.0) / 2.0 * 255.0).round().astype(np.uint8)

        return {
            "rgb_uint8": rgb_uint8,
            "rgb_float": rgb_float,
            "weight_map": weight_accum,
            "tiles_processed": num_tiles,
            "scene_shape": (H, W),
            "meta": meta,
        }

    def save_output(
        self,
        result: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """
        Save synthesized RGB scene to disk (GeoTIFF, TIFF, or PNG).
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_uint8 = result["rgb_uint8"]
        meta = result.get("meta", {})

        if out_path.suffix.lower() in (".tif", ".tiff") and rasterio is not None and meta.get("crs"):
            # Preserve GeoTIFF georeferencing
            out_meta = meta.copy()
            out_meta.update(
                {
                    "count": 3,
                    "dtype": "uint8",
                    "nodata": None,
                    "height": rgb_uint8.shape[0],
                    "width": rgb_uint8.shape[1],
                }
            )
            with rasterio.open(out_path, "w", **out_meta) as dst:
                for c in range(3):
                    dst.write(rgb_uint8[:, :, c], c + 1)
        elif out_path.suffix.lower() in (".tif", ".tiff") and tifffile is not None:
            tifffile.imwrite(out_path, rgb_uint8)
        else:
            Image.fromarray(rgb_uint8).save(out_path)

        return out_path
