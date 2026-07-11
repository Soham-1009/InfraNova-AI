from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from demo.utils import postprocess_output, preprocess_ir_image
from split_patches import split_regions
from src.utils.image_processing import to_single_band_array
from src.utils.logger import TrainingLogger


class ImageProcessingTests(unittest.TestCase):
    def test_preserves_16_bit_pil_values(self) -> None:
        source = np.array([[0, 256, 1024, 4095]], dtype=np.uint16)
        image = Image.fromarray(source)

        actual = to_single_band_array(image)

        np.testing.assert_array_equal(actual, source.astype(np.float32))

    def test_converts_channel_first_rgb_to_luminance(self) -> None:
        source = np.array([[[100]], [[200]], [[50]]], dtype=np.uint8)

        actual = to_single_band_array(source)

        self.assertEqual(actual.shape, (1, 1))
        self.assertAlmostEqual(float(actual[0, 0]), 153.0, places=2)

    def test_preprocess_returns_model_ready_tensor(self) -> None:
        source = np.arange(64, dtype=np.uint16).reshape(8, 8) * 64

        tensor = preprocess_ir_image(Image.fromarray(source), image_size=16)

        self.assertEqual(tuple(tensor.shape), (1, 1, 16, 16))
        self.assertTrue(torch.all(tensor >= -1.0))
        self.assertTrue(torch.all(tensor <= 1.0))

    def test_postprocess_rejects_multi_image_batch(self) -> None:
        with self.assertRaises(ValueError):
            postprocess_output(torch.zeros(2, 3, 4, 4))


class PipelineRobustnessTests(unittest.TestCase):
    def test_small_region_split_keeps_training_and_validation_usable(self) -> None:
        regions = [Path(f"region_{index}") for index in range(3)]

        result = split_regions(regions, seed=42)

        self.assertEqual(len(result["train"]), 1)
        self.assertEqual(len(result["val"]), 1)
        self.assertEqual(len(result["test"]), 1)

    def test_logger_accepts_metrics_added_later(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrainingLogger(log_dir=temp_dir)
            logger.log_epoch(1, {"g_loss": 1.0})
            logger.log_epoch(2, {"g_loss": 0.5, "val_ssim": 0.3})

            with (Path(temp_dir) / "training.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["val_ssim"], "")
        self.assertEqual(rows[1]["val_ssim"], "0.3")


if __name__ == "__main__":
    unittest.main()
