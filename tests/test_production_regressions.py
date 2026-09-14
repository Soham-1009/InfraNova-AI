from __future__ import annotations

import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main
from scripts.deployment import model_summary
from scripts.deployment.batch_inference import build_output_stems
from scripts.deployment.export import export_model
from scripts.evaluation.benchmark import benchmark_device
from scripts.evaluation.validate_dataset import validate_sample
from src.inference.landsat_inference import LandsatColorizationInference
from src.models.pix2pix.pix2pix import Pix2Pix
from src.training.trainer import Trainer
from src.utils.checkpoint import load_pix2pix_model
from src.utils.config_validator import validate_config


class TinyPix2Pix(nn.Module):
    """Small GAN-shaped model used to exercise trainer error handling quickly."""

    def __init__(self) -> None:
        super().__init__()
        self.generator = nn.Conv2d(2, 3, kernel_size=1)
        self.discriminator = nn.Conv2d(5, 1, kernel_size=1)
        self.multi_scale = False

    def generate(self, ir: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.generator(ir))

    def discriminate(self, ir: torch.Tensor, rgb: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        output = self.discriminator(torch.cat([ir, rgb], dim=1))
        return (output, [output]) if return_features else output


def _trainer_config(tmp_path: Path, grad_clip: float | None) -> dict:
    return {
        "device": "cpu",
        "training": {
            "grad_clip": grad_clip,
            "epochs": 1,
            "amp": False,
            "optimizer": {"lr": 1e-4, "beta1": 0.5, "beta2": 0.999},
        },
        "loss": {
            "lambda_adv": 1.0,
            "lambda_l1": 1.0,
            "lambda_perc": 0.0,
            "lambda_ssim": 0.0,
            "lambda_chroma": 0.0,
            "lambda_sat": 0.0,
            "lambda_feat": 0.0,
        },
        "paths": {"checkpoints": str(tmp_path), "outputs": str(tmp_path), "logs": str(tmp_path)},
        "logging": {"use_wandb": False},
    }


def test_pix2pix_device_cache_tracks_to_empty() -> None:
    model = Pix2Pix(device="meta", generator_impl="resnet")

    model.to_empty(device="cpu")

    assert model.device.type == "cpu"
    assert next(model.generator.parameters()).device.type == "cpu"


def test_strict_loader_rejects_partial_checkpoint(tmp_path: Path) -> None:
    model = Pix2Pix(device="cpu", generator_impl="resnet")
    checkpoint_path = tmp_path / "partial.pth"
    torch.save(
        {
            "arch_info": {
                "generator_impl": "resnet",
                "input_channels": 2,
                "output_channels": 3,
                "image_size": 128,
                "discriminator_scales": 2,
            },
            "model_state_dict": {"generator.model.1.weight": model.state_dict()["generator.model.1.weight"]},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="strictly"):
        load_pix2pix_model(checkpoint_path, device="cpu")


def test_source_inference_preprocesses_two_thermal_bands() -> None:
    engine = LandsatColorizationInference(checkpoint_path="unused.pth", device="cpu")
    band10 = np.arange(64, dtype=np.float32).reshape(8, 8)
    band11 = band10 + 20.0

    tensor = engine.preprocess((band10, band11))

    assert tuple(tensor.shape) == (1, 2, 128, 128)
    assert torch.isfinite(tensor).all()


def test_invalid_clahe_grid_returns_client_error() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")

    response = TestClient(api_main.app).post(
        "/postprocess/clahe?grid_size=0",
        files={"file": ("input.png", buf.getvalue(), "image/png")},
    )

    assert response.status_code == 400
    assert "grid_size" in response.json()["detail"]


def test_engine_singleton_retries_after_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakyEngine:
        load_attempts = 0

        def __init__(self, *args, **kwargs) -> None:
            self.model = None
            self.device = torch.device("cpu")

        def load_model(self) -> None:
            type(self).load_attempts += 1
            if type(self).load_attempts == 1:
                raise RuntimeError("simulated checkpoint error")
            self.model = object()

    previous_engine = api_main.engine
    api_main.engine = None
    monkeypatch.setattr(api_main, "InferenceEngine", FlakyEngine)
    try:
        with pytest.raises(RuntimeError, match="simulated checkpoint error"):
            api_main.get_engine()
        assert api_main.engine is None

        loaded = api_main.get_engine()
        assert loaded.model is not None
        assert FlakyEngine.load_attempts == 2
    finally:
        api_main.engine = previous_engine


def test_engine_singleton_initializes_once_under_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowEngine:
        load_attempts = 0

        def __init__(self, *args, **kwargs) -> None:
            self.model = None
            self.device = torch.device("cpu")

        def load_model(self) -> None:
            type(self).load_attempts += 1
            time.sleep(0.02)
            self.model = object()

    previous_engine = api_main.engine
    api_main.engine = None
    monkeypatch.setattr(api_main, "InferenceEngine", SlowEngine)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            engines = list(pool.map(lambda _: api_main.get_engine(), range(4)))
        assert SlowEngine.load_attempts == 1
        assert len({id(engine) for engine in engines}) == 1
    finally:
        api_main.engine = previous_engine


def test_trainer_allows_none_gradient_clip(tmp_path: Path) -> None:
    trainer = Trainer(TinyPix2Pix(), [], [], _trainer_config(tmp_path, grad_clip=None))

    assert trainer.grad_clip is None


def test_trainer_skips_nonfinite_loss_before_backward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = {"ir": torch.zeros(1, 2, 16, 16), "rgb": torch.zeros(1, 3, 16, 16)}
    trainer = Trainer(TinyPix2Pix(), [batch], [], _trainer_config(tmp_path, grad_clip=None))
    backward_calls: list[torch.Tensor] = []
    nonfinite_loss = torch.tensor(float("nan"), requires_grad=True)
    nonfinite_loss.register_hook(lambda gradient: backward_calls.append(gradient))
    monkeypatch.setattr(trainer, "_disc_loss_multi_scale", lambda *args, **kwargs: nonfinite_loss)

    metrics = trainer.train_one_epoch(0)

    assert not backward_calls
    assert math.isfinite(metrics["d_loss"])


def test_config_validator_rejects_nonfinite_numeric_values(tmp_path: Path) -> None:
    config = {
        "project": {"name": "test"},
        "dataset": {"root_dir": str(tmp_path), "image_size": 128, "input_channels": 2, "output_channels": 3},
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "decay_start_epoch": 0,
            "patience": 1,
            "grad_clip": None,
            "optimizer": {"lr": float("nan"), "beta1": 0.5, "beta2": 0.999},
        },
        "model": {"generator": {"implementation": "hd"}},
        "paths": {"checkpoints": str(tmp_path), "logs": str(tmp_path)},
    }

    with pytest.raises(ValueError, match=r"optimizer\.lr"):
        validate_config(config)


def test_validator_requires_band11_for_dual_band_dataset(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    np.save(sample_dir / "tir_200m.npy", np.random.rand(64, 64).astype(np.float32))
    np.save(sample_dir / "tir_100m.npy", np.random.rand(128, 128).astype(np.float32))
    np.save(sample_dir / "rgb_100m.npy", np.random.rand(3, 128, 128).astype(np.float32))

    issues, _hashes = validate_sample(sample_dir)

    assert "Missing tir_b11_100m.npy" in issues


def test_recursive_batch_output_stems_are_unique(tmp_path: Path) -> None:
    first = tmp_path / "north" / "scene.tif"
    second = tmp_path / "south" / "scene.tif"
    third = tmp_path / "north" / "scene.png"
    stems = build_output_stems([first, second, third], tmp_path)

    assert stems[first] == Path("north") / "scene"
    assert stems[second] == Path("south") / "scene"
    assert len({str(stem).casefold() for stem in stems.values()}) == 3


def test_export_script_uses_repository_root() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert project_root == export_model.PROJECT_ROOT


def test_model_summary_uses_checkpoint_architecture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class SummaryModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = nn.Conv2d(2, 3, kernel_size=1)
            self.discriminator = nn.Conv2d(5, 1, kernel_size=1)
            self.out_channels = 3

    monkeypatch.setattr(
        model_summary,
        "load_pix2pix_model",
        lambda *args, **kwargs: (SummaryModel(), {"input_channels": 2, "image_size": 128}),
    )

    summary = model_summary.generate_summary(
        checkpoint_path="checkpoint.pth", output_path=str(tmp_path / "summary.txt")
    )

    assert "Input:  2 x 128 x 128" in summary
    assert "Output: 3 x 128 x 128" in summary


def test_integration_test_uses_its_checkout_root() -> None:
    from tests import test_training_integration

    assert Path(__file__).resolve().parents[1] == test_training_integration.PROJECT_ROOT


def test_benchmark_rejects_zero_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        benchmark_device(nn.Identity(), "cpu", 128, 2, warmup=0, iterations=0)


def test_inference_docs_identify_the_active_exp9_checkpoint() -> None:
    project_root = Path(__file__).resolve().parents[1]
    docs = (project_root / "docs" / "inference" / "INFERENCE.md").read_text(encoding="utf-8")

    assert "**Active Production (Exp9)**" in docs
    assert "71bbda3f31b85e7e741b26d5ce0ff398a0394c7dd4ef6ff5452f31f7e0400382" in docs
    assert "--generator_type" not in docs
