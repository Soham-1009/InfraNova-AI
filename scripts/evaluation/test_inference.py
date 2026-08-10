import argparse
import logging

import torch
import yaml

from src.datasets.landsat9_dataset import Landsat9Dataset
from src.models.pix2pix.pix2pix import Pix2Pix

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_inference_pass(model: Pix2Pix, ir_tensor: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        output = model.generate(ir_tensor)
    return output

def validate_tensor(tensor: torch.Tensor, name: str) -> None:
    logger.info(f"Validating {name}...")
    logger.info(f"  Shape: {tensor.shape}")
    logger.info(f"  Dtype: {tensor.dtype}")

    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaNs or Infs!")

    min_val, max_val = tensor.min().item(), tensor.max().item()
    logger.info(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")

    if min_val < -2.0 or max_val > 2.0:
        logger.warning(f"{name} values ({min_val:.4f}, {max_val:.4f}) seem heavily out of bounds for normalized data.")
    else:
        logger.info(f"  {name} validation passed.")

def test_inference_consistency(config_path: str, checkpoint_path: str):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("Starting Standalone Inference & Reload Consistency Test")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(config_path)

    # 1. Prepare a single data sample deterministically
    dataset = Landsat9Dataset(
        root_dir=cfg["dataset"]["root_dir"],
        split="val",
        image_size=int(cfg["dataset"]["image_size"]),
        augment=False,
        normalization=cfg["dataset"]["normalization"]["mode"],
    )

    sample = dataset[0]
    ir_tensor = sample["ir"].unsqueeze(0).to(device)
    logger.info(f"Loaded validation sample 0. IR Tensor shape: {ir_tensor.shape}")

    # 2. Build model and load checkpoint (PASS 1)
    logger.info("\n--- PASS 1: Loading Checkpoint ---")
    multi_scale = bool(cfg.get("model", {}).get("multi_scale_disc", False))
    model_pass1 = Pix2Pix(
        device=device,
        in_channels=int(cfg["dataset"]["input_channels"]),
        out_channels=int(cfg["dataset"]["output_channels"]),
        image_size=int(cfg["dataset"]["image_size"]),
        multi_scale=multi_scale,
        generator_impl=cfg.get("model", {}).get("generator", {}).get("implementation", "dynamic"),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Strip ".module." from state dict keys (caused by Kaggle multi-GPU training)
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {k.replace(".module.", "."): v for k, v in state_dict.items()}
    model_pass1.load_state_dict(clean_state_dict)

    output_pass1 = run_inference_pass(model_pass1, ir_tensor)
    validate_tensor(output_pass1, "Output Pass 1")

    # 3. Build model and load checkpoint (PASS 2)
    logger.info("\n--- PASS 2: Reloading Checkpoint ---")
    model_pass2 = Pix2Pix(
        device=device,
        in_channels=int(cfg["dataset"]["input_channels"]),
        out_channels=int(cfg["dataset"]["output_channels"]),
        image_size=int(cfg["dataset"]["image_size"]),
        multi_scale=multi_scale,
        generator_impl=cfg.get("model", {}).get("generator", {}).get("implementation", "dynamic"),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {k.replace(".module.", "."): v for k, v in state_dict.items()}
    model_pass2.load_state_dict(clean_state_dict)

    output_pass2 = run_inference_pass(model_pass2, ir_tensor)
    validate_tensor(output_pass2, "Output Pass 2")

    # 4. Assert consistency
    logger.info("\n--- Validating Consistency ---")
    diff = torch.abs(output_pass1 - output_pass2)
    max_diff = diff.max().item()
    logger.info(f"Maximum absolute difference between Pass 1 and Pass 2: {max_diff:.8e}")

    # Assert using standard floating point tolerance
    if not torch.allclose(output_pass1, output_pass2, atol=1e-6, rtol=1e-5):
        raise AssertionError(f"Reload consistency failed! Max diff {max_diff} exceeds tolerance.")

    logger.info("SUCCESS: Checkpoint reload consistency verified (outputs match within floating-point tolerance).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    test_inference_consistency(args.config, args.checkpoint)
