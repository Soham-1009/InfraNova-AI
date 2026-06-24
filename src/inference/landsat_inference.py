from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.models.pix2pix.pix2pix import Pix2Pix

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class LandsatColorizationInference:
    """
    Production inference engine for Landsat 9 TIR -> RGB colorization.

    Features:
        - PIL or NumPy inputs
        - Percentile stretching for TIR
        - Optional TTA
        - TIFF export with band order B, G, R (Layer 1, 2, 3)
        - Heuristic confidence score
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best/pix2pix_landsat_best.pth",
        device: Optional[str] = None,
        image_size: int = 256,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = int(image_size)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.model: Optional[Pix2Pix] = None

    def load_model(self) -> Pix2Pix:
        """
        Load trained Pix2Pix model from checkpoint.
        """
        if self.model is not None:
            return self.model

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        model = Pix2Pix(device=self.device, in_channels=1, out_channels=3)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            # Fallback for generator-only checkpoints
            model.generator.load_state_dict(checkpoint["generator_state_dict"], strict=True)
            model.eval()
            self.model = model
            return self.model
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model
        logger.info("Loaded model from %s on %s", self.checkpoint_path, self.device)
        return self.model

    @staticmethod
    def _to_grayscale_array(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Convert PIL / NumPy input to a 2D grayscale NumPy array.
        """
        if isinstance(image, Image.Image):
            return np.array(image.convert("L"))

        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                return arr
            if arr.ndim == 3 and arr.shape[2] >= 1:
                return arr[:, :, 0]
            raise ValueError(f"Unsupported NumPy image shape: {arr.shape}")

        raise TypeError("Input must be PIL.Image.Image or numpy.ndarray")

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Resize to 256x256 and normalize TIR with percentile stretching.
        """
        arr = self._to_grayscale_array(image).astype(np.float32)

        # Percentile stretching per-image for thermal contrast normalization
        lo = np.percentile(arr, self.percentile_low)
        hi = np.percentile(arr, self.percentile_high)

        if hi - lo < 1e-6:
            logger.warning("Flat input image detected; returning zeros after normalization.")
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo)

        arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        arr = arr * 2.0 - 1.0  # [-1, 1]

        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return tensor

    @staticmethod
    def _denormalize_rgb(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert model output tensor in [-1, 1] to uint8 RGB NumPy array.
        """
        tensor = tensor.detach().cpu().clamp(-1, 1)
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()
        tensor = (tensor * 255.0).round().astype(np.uint8)
        return tensor

    @staticmethod
    def _confidence_from_tta(predictions: np.ndarray) -> float:
        """
        Confidence heuristic from TTA agreement.
        Lower inter-augmentation variance => higher confidence.
        """
        if predictions.ndim != 4:
            return 0.0

        std_map = predictions.std(axis=0)  # [3, H, W]
        mean_std = float(std_map.mean())
        confidence = float(np.exp(-8.0 * mean_std))
        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _confidence_from_output(rgb: np.ndarray) -> float:
        """
        Confidence heuristic when TTA is disabled.
        Uses luminance contrast and saturation balance as a proxy.
        """
        rgb_f = rgb.astype(np.float32) / 255.0
        luminance = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
        contrast = float(luminance.std())

        # Normalize contrast to a rough 0..1 score
        confidence = np.clip(contrast / 0.25, 0.0, 1.0)
        return float(confidence)

    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
        use_tta: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference and return RGB output plus confidence score.

        Returns:
            {
                "rgb": PIL.Image (RGB),
                "rgb_array": np.ndarray uint8 [H, W, 3],
                "tiff_bgr": np.ndarray uint8 [H, W, 3],  # band order B, G, R
                "confidence": float
            }
        """
        model = self.load_model()
        x = self.preprocess(image).to(self.device)

        if not use_tta:
            y = model.generate(x)
            rgb = self._denormalize_rgb(y)
            confidence = self._confidence_from_output(rgb)
        else:
            preds = []

            # Identity
            y0 = model.generate(x)
            preds.append(y0.squeeze(0).detach().cpu().numpy())

            # Horizontal flip
            x_h = torch.flip(x, dims=[3])
            y_h = model.generate(x_h)
            y_h = torch.flip(y_h, dims=[3])
            preds.append(y_h.squeeze(0).detach().cpu().numpy())

            # Vertical flip
            x_v = torch.flip(x, dims=[2])
            y_v = model.generate(x_v)
            y_v = torch.flip(y_v, dims=[2])
            preds.append(y_v.squeeze(0).detach().cpu().numpy())

            # 180 rotation
            x_r = torch.flip(x, dims=[2, 3])
            y_r = model.generate(x_r)
            y_r = torch.flip(y_r, dims=[2, 3])
            preds.append(y_r.squeeze(0).detach().cpu().numpy())

            preds_np = np.stack(preds, axis=0)  # [N, 3, H, W]
            y_mean = torch.from_numpy(preds_np.mean(axis=0)).unsqueeze(0)
            rgb = self._denormalize_rgb(y_mean)
            confidence = self._confidence_from_tta(preds_np)

        pil_rgb = Image.fromarray(rgb, mode="RGB")
        bgr_for_tiff = rgb[:, :, ::-1].copy()  # Layer 1=B, Layer 2=G, Layer 3=R

        return {
            "rgb": pil_rgb,
            "rgb_array": rgb,
            "tiff_bgr": bgr_for_tiff,
            "confidence": confidence,
        }

    @staticmethod
    def save_tiff(bgr_array: np.ndarray, output_path: str) -> str:
        """
        Save a 3-band TIFF with band order B, G, R (Layer 1, 2, 3).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if bgr_array.ndim != 3 or bgr_array.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 array, got {bgr_array.shape}")

        if rasterio is not None:
            height, width, _ = bgr_array.shape
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=3,
                dtype=bgr_array.dtype,
            ) as dst:
                # Store in exact order: band 1=B, band 2=G, band 3=R
                dst.write(bgr_array[:, :, 0], 1)
                dst.write(bgr_array[:, :, 1], 2)
                dst.write(bgr_array[:, :, 2], 3)
        else:
            # Fallback: PIL save. Band order is preserved in the array, but
            # raster semantics depend on the reader. Rasterio is preferred.
            Image.fromarray(bgr_array[:, :, ::-1], mode="RGB").save(path)

        return str(path)

    def predict_and_save(
        self,
        image: Union[Image.Image, np.ndarray],
        output_path: str,
        use_tta: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict and save a TIFF output.
        """
        result = self.predict(image=image, use_tta=use_tta)
        self.save_tiff(result["tiff_bgr"], output_path)
        return result