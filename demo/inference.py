from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

# Make sure project root is importable when running from /demo
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_torch_checkpoint
from demo.utils import postprocess_output, preprocess_ir_image


class InferenceEngine:
    """
    Inference engine for InfraNova AI Pix2Pix model.

    Supports:
    - CPU and GPU execution
    - PIL.Image and numpy.ndarray inputs
    - Single-image and batch inference
    - Optional test-time augmentation (TTA)
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best/pix2pix_landsat_best.pth",
        device: Optional[str] = None,
        image_size: int = 256,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = int(image_size)
        if self.image_size < 256 or self.image_size % 256 != 0:
            raise ValueError("image_size must be a multiple of 256 for this generator")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[Pix2Pix] = None

    def load_model(self) -> Pix2Pix:
        """
        Load Pix2Pix checkpoint into memory.

        Returns:
            Loaded Pix2Pix model in eval mode.
        """
        if self.model is not None:
            return self.model

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Create model on selected device
        model = Pix2Pix(device=self.device)

        checkpoint = load_torch_checkpoint(self.checkpoint_path, map_location=self.device)

        # Support common checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            # Fallback if only generator is saved separately
            state_dict = checkpoint["generator_state_dict"]
            model.generator.load_state_dict(state_dict, strict=True)
            self.model = model.eval()
            return self.model
        else:
            # Assume raw model state_dict
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        return self.model

    @staticmethod
    def _to_pil_or_array(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Convert PIL or numpy input to PIL grayscale image.

        Preserves original bit depth for thermal data instead of
        truncating to uint8.

        Args:
            image: Input image.

        Returns:
            PIL Image.
        """
        if isinstance(image, Image.Image):
            # Keep original mode — avoid .convert("L") which truncates 16-bit to 8-bit
            if image.mode in ("I;16", "I;16B", "I"):
                return image
            return image.convert("L")

        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                return Image.fromarray(arr)
            if arr.ndim == 3:
                # Channel-first: e.g. (3, 128, 128) or (1, 128, 128)
                if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
                    if arr.shape[0] == 1:
                        return Image.fromarray(arr[0])
                    # Multi-channel first — convert via luminance
                    lum = (
                        0.299 * arr[0].astype(np.float32)
                        + 0.587 * arr[1].astype(np.float32)
                        + 0.114 * arr[2].astype(np.float32)
                    )
                    return Image.fromarray(lum.astype(arr.dtype))
                # Channel-last: e.g. (128, 128, 3) or (128, 128, 1)
                if arr.shape[2] == 3:
                    # RGB — convert via luminance
                    lum = (
                        0.299 * arr[..., 0].astype(np.float32)
                        + 0.587 * arr[..., 1].astype(np.float32)
                        + 0.114 * arr[..., 2].astype(np.float32)
                    )
                    return Image.fromarray(lum.astype(arr.dtype))
                if arr.shape[2] == 1:
                    return Image.fromarray(arr[:, :, 0])

        raise TypeError("Input must be a PIL.Image.Image or numpy.ndarray.")

    @torch.inference_mode()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
        use_tta: bool = False,
    ) -> Image.Image:
        """
        Predict RGB output from a single IR image.

        Args:
            image: PIL image or numpy array.
            use_tta: Enable simple flip/rotation TTA.

        Returns:
            PIL RGB image.
        """
        model = self.load_model()
        ir_tensor = preprocess_ir_image(image, image_size=self.image_size).to(self.device)

        if not use_tta:
            fake_rgb = model.generate(ir_tensor)
            return postprocess_output(fake_rgb.squeeze(0))

        # TTA: average predictions across a small set of geometric transforms.
        # For each transform, we invert the transform on the output before averaging.
        preds: List[torch.Tensor] = []

        # Original
        preds.append(model.generate(ir_tensor))

        # Horizontal flip
        ir_h = torch.flip(ir_tensor, dims=[3])
        pred_h = model.generate(ir_h)
        pred_h = torch.flip(pred_h, dims=[3])
        preds.append(pred_h)

        # Vertical flip
        ir_v = torch.flip(ir_tensor, dims=[2])
        pred_v = model.generate(ir_v)
        pred_v = torch.flip(pred_v, dims=[2])
        preds.append(pred_v)

        # 180-degree rotation (flip both axes)
        ir_r = torch.flip(ir_tensor, dims=[2, 3])
        pred_r = model.generate(ir_r)
        pred_r = torch.flip(pred_r, dims=[2, 3])
        preds.append(pred_r)

        fake_rgb = torch.stack(preds, dim=0).mean(dim=0)
        return postprocess_output(fake_rgb.squeeze(0))

    @torch.inference_mode()
    def predict_batch(
        self,
        images: Sequence[Union[Image.Image, np.ndarray]],
        use_tta: bool = False,
    ) -> List[Image.Image]:
        """
        Predict RGB outputs for a batch of IR images.

        Args:
            images: Sequence of PIL images or numpy arrays.
            use_tta: Enable TTA per image.

        Returns:
            List of PIL RGB images.
        """
        outputs: List[Image.Image] = []
        for img in images:
            outputs.append(self.predict(img, use_tta=use_tta))
        return outputs
