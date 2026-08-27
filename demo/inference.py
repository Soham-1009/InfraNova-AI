from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make sure project root is importable when running from /demo
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.utils import postprocess_output, preprocess_ir_image
from src.models.pix2pix.pix2pix import Pix2Pix
from src.utils.checkpoint import load_torch_checkpoint


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
        checkpoint_path: str = "outputs/best/pix2pix_landsat_best.pth",
        device: str | None = None,
        image_size: int = 128,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            alt_path = Path("checkpoints/best/pix2pix_landsat_best.pth")
            if alt_path.exists():
                self.checkpoint_path = alt_path
        self.image_size = int(image_size)
        if self.image_size < 128 or self.image_size % 128 != 0:
            raise ValueError("image_size must be a multiple of 128 for this generator")
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Pix2Pix | None = None

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

        checkpoint = load_torch_checkpoint(self.checkpoint_path, map_location=self.device)

        gen_impl = "dynamic"
        num_scales = 1
        in_channels = 1

        state = {}
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            state = checkpoint["generator_state_dict"]
        elif isinstance(checkpoint, dict):
            state = checkpoint

        for k, v in state.items():
            if "global_gen" in k or "local_enhancer" in k or "global_generator" in k:
                gen_impl = "hd"
                num_scales = 2
            if (
                "global_gen.downs.0.block.0.weight" in k
                or "global_generator.model.0.weight" in k
                or "down1.model.0.weight" in k
                or "generator.down1.model.0.weight" in k
            ):
                in_channels = v.shape[1]

        if isinstance(checkpoint, dict) and "arch_info" in checkpoint:
            arch_info = checkpoint["arch_info"]
            gen_impl = arch_info.get("generator_impl", gen_impl)
            num_scales = arch_info.get("discriminator_scales", num_scales)
            in_channels = arch_info.get("in_channels", in_channels)

        self.in_channels = in_channels

        # Create model on selected device
        model = Pix2Pix(
            device=self.device,
            in_channels=in_channels,
            out_channels=3,
            generator_impl=gen_impl,
            image_size=self.image_size,
            num_scales=num_scales,
        )

        if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            clean_gen = {k.replace("module.", ""): v for k, v in checkpoint["generator_state_dict"].items()}
            model.generator.load_state_dict(clean_gen, strict=True)
            self.model = model.eval()
            return self.model

        clean_state_dict = {k.replace(".module.", ".").replace("module.", ""): v for k, v in state.items()}
        model_dict = model.state_dict()
        matched = {k: v for k, v in clean_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(matched, strict=False)
        model.eval()

        self.model = model
        return self.model

    @staticmethod
    def _to_pil_or_array(image: Image.Image | np.ndarray) -> Image.Image:
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
        image: Image.Image | np.ndarray,
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
        ir_tensor = preprocess_ir_image(
            image,
            image_size=self.image_size,
            target_channels=getattr(self, "in_channels", 2),
        ).to(self.device)

        if not use_tta:
            fake_rgb = model.generate(ir_tensor)
            return postprocess_output(fake_rgb.squeeze(0))

        # TTA: average predictions across a small set of geometric transforms.
        # For each transform, we invert the transform on the output before averaging.
        preds: list[torch.Tensor] = []

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
        images: Sequence[Image.Image | np.ndarray],
        use_tta: bool = False,
    ) -> list[Image.Image]:
        """
        Predict RGB outputs for a batch of IR images.

        Args:
            images: Sequence of PIL images or numpy arrays.
            use_tta: Enable TTA per image.

        Returns:
            List of PIL RGB images.
        """
        if not images:
            return []

        model = self.load_model()
        tensors = [preprocess_ir_image(img, image_size=self.image_size) for img in images]
        batch_tensor = torch.cat(tensors, dim=0).to(self.device)

        if not use_tta:
            fake_rgb = model.generate(batch_tensor)
        else:
            preds: list[torch.Tensor] = []
            preds.append(model.generate(batch_tensor))

            batch_h = torch.flip(batch_tensor, dims=[3])
            pred_h = model.generate(batch_h)
            preds.append(torch.flip(pred_h, dims=[3]))

            batch_v = torch.flip(batch_tensor, dims=[2])
            pred_v = model.generate(batch_v)
            preds.append(torch.flip(pred_v, dims=[2]))

            batch_r = torch.flip(batch_tensor, dims=[2, 3])
            pred_r = model.generate(batch_r)
            preds.append(torch.flip(pred_r, dims=[2, 3]))

            fake_rgb = torch.stack(preds, dim=0).mean(dim=0)

        outputs: list[Image.Image] = []
        for i in range(fake_rgb.size(0)):
            outputs.append(postprocess_output(fake_rgb[i]))
        return outputs
