"""
Automatic ablation study for InfraNova AI Phase 3 improvements.

Generates config variants and a run plan for systematic comparison.
Can either generate configs only (safe) or run full training orchestration.

Usage:
    python run_ablation_study.py --generate       # Generate configs only
    python run_ablation_study.py --run             # Generate + run all sequentially
    python run_ablation_study.py --help
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "baseline",
        "description": "Original Pix2Pix with local normalization and original loss weights",
        "overrides": {
            "dataset.normalization.mode": "local",
            "loss.lambda_l1": 50.0,
            "loss.lambda_perc": 20.0,
            "loss.lambda_ssim": 3.0,
            "loss.lambda_chroma": 0.0,
            "loss.lambda_feat": 0.0,
            "loss.gan_mode": "bce",
            "model.multi_scale_disc": False,
        },
    },
    {
        "name": "global_norm",
        "description": "Global normalization only",
        "overrides": {
            "dataset.normalization.mode": "global",
            "dataset.normalization.stats_file": "configs/normalization_stats.json",
            "loss.lambda_l1": 50.0,
            "loss.lambda_perc": 20.0,
            "loss.lambda_ssim": 3.0,
            "loss.lambda_chroma": 0.0,
            "loss.lambda_feat": 0.0,
            "loss.gan_mode": "bce",
        },
    },
    {
        "name": "loss_tuning",
        "description": "Rebalanced loss weights (lower L1, higher SSIM)",
        "overrides": {
            "dataset.normalization.mode": "global",
            "dataset.normalization.stats_file": "configs/normalization_stats.json",
            "loss.lambda_l1": 10.0,
            "loss.lambda_perc": 10.0,
            "loss.lambda_ssim": 5.0,
            "loss.lambda_chroma": 0.0,
            "loss.lambda_feat": 0.0,
            "loss.gan_mode": "bce",
        },
    },
    {
        "name": "chroma_loss",
        "description": "Loss tuning + chroma loss for color vibrancy",
        "overrides": {
            "dataset.normalization.mode": "global",
            "dataset.normalization.stats_file": "configs/normalization_stats.json",
            "loss.lambda_l1": 10.0,
            "loss.lambda_perc": 10.0,
            "loss.lambda_ssim": 5.0,
            "loss.lambda_chroma": 2.0,
            "loss.lambda_feat": 0.0,
            "loss.gan_mode": "bce",
        },
    },
    {
        "name": "feat_matching",
        "description": "Chroma + feature matching loss",
        "overrides": {
            "dataset.normalization.mode": "global",
            "dataset.normalization.stats_file": "configs/normalization_stats.json",
            "loss.lambda_l1": 10.0,
            "loss.lambda_perc": 10.0,
            "loss.lambda_ssim": 5.0,
            "loss.lambda_chroma": 2.0,
            "loss.lambda_feat": 10.0,
            "loss.gan_mode": "bce",
        },
    },
    {
        "name": "lsgan",
        "description": "All improvements + LSGAN loss",
        "overrides": {
            "dataset.normalization.mode": "global",
            "dataset.normalization.stats_file": "configs/normalization_stats.json",
            "loss.lambda_l1": 10.0,
            "loss.lambda_perc": 10.0,
            "loss.lambda_ssim": 5.0,
            "loss.lambda_chroma": 2.0,
            "loss.lambda_feat": 10.0,
            "loss.gan_mode": "lsgan",
        },
    },
    {
        "name": "cosine_sched",
        "description": "All improvements + cosine LR scheduler",
        "overrides": {
            "dataset.normalization.mode": "global",
            "dataset.normalization.stats_file": "configs/normalization_stats.json",
            "loss.lambda_l1": 10.0,
            "loss.lambda_perc": 10.0,
            "loss.lambda_ssim": 5.0,
            "loss.lambda_chroma": 2.0,
            "loss.lambda_feat": 10.0,
            "loss.gan_mode": "lsgan",
            "scheduler.type": "cosine",
        },
    },
    {
        "name": "multi_scale",
        "description": "Full improvements + multi-scale discriminator",
        "overrides": {
            "dataset.normalization.mode": "global",
            "dataset.normalization.stats_file": "configs/normalization_stats.json",
            "loss.lambda_l1": 10.0,
            "loss.lambda_perc": 10.0,
            "loss.lambda_ssim": 5.0,
            "loss.lambda_chroma": 2.0,
            "loss.lambda_feat": 10.0,
            "loss.gan_mode": "lsgan",
            "scheduler.type": "cosine",
            "model.multi_scale_disc": True,
        },
    },
]


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dotted key notation."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def generate_configs(
    base_config_path: str = str(PROJECT_ROOT / "configs/config.yaml"),
    output_dir: str = "configs/ablation",
    epochs: int = 200,
) -> list[dict[str, Any]]:
    """Generate ablation study config files."""
    with open(base_config_path, encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generated = []
    for exp in EXPERIMENTS:
        cfg = copy.deepcopy(base_config)
        cfg["project"]["name"] = f"InfraNova-{exp['name']}"
        cfg["training"]["epochs"] = epochs
        cfg["training"]["resume_from"] = ""  # Start fresh for ablation

        for key, value in exp["overrides"].items():
            _set_nested(cfg, key, value)

        config_path = out_path / f"config_{exp['name']}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        generated.append({
            "name": exp["name"],
            "description": exp["description"],
            "config_path": str(config_path),
        })
        print(f"  Generated: {config_path}")

    # Save run plan
    plan_path = out_path / "ablation_plan.json"
    plan_path.write_text(
        json.dumps(generated, indent=2), encoding="utf-8"
    )
    print(f"\nRun plan saved to: {plan_path}")

    return generated


def run_experiments(configs: list[dict[str, Any]]) -> None:
    """Run all ablation experiments sequentially."""
    from src.training.train_landsat import load_config, run_training

    results = []
    for exp in configs:
        print(f"\n{'=' * 60}")
        print(f"Running: {exp['name']} — {exp['description']}")
        print(f"Config: {exp['config_path']}")
        print(f"{'=' * 60}")

        try:
            cfg = load_config(exp["config_path"])
            history = run_training(cfg)

            # Extract best metrics
            val_ssim = history.get("val_ssim", [])
            val_psnr = history.get("val_psnr", [])

            results.append({
                "experiment": exp["name"],
                "description": exp["description"],
                "best_ssim": max(val_ssim) if val_ssim else 0.0,
                "best_psnr": max(val_psnr) if val_psnr else 0.0,
                "epochs_trained": len(val_ssim),
                "status": "completed",
            })
        except Exception as e:
            print(f"ERROR in {exp['name']}: {e}")
            results.append({
                "experiment": exp["name"],
                "description": exp["description"],
                "best_ssim": 0.0,
                "best_psnr": 0.0,
                "epochs_trained": 0,
                "status": f"failed: {e}",
            })

    # Save results
    csv_path = Path("outputs/ablation_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAblation results saved to: {csv_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Ablation Study Summary")
    print(f"{'=' * 60}")
    for r in results:
        print(
            f"  {r['experiment']:>15s} | SSIM={r['best_ssim']:.4f} | "
            f"PSNR={r['best_psnr']:.2f} | {r['status']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run InfraNova AI Phase 3 ablation study."
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate config files only (safe, no training).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Generate configs and run all experiments sequentially.",
    )
    parser.add_argument(
        "--base-config",
        default=str(PROJECT_ROOT / "configs/config.yaml"),
        help="Base config to derive ablation variants from.",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/ablation",
        help="Directory for generated config files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs per ablation experiment.",
    )
    args = parser.parse_args()

    if not args.generate and not args.run:
        parser.print_help()
        print("\nPlease specify --generate or --run.")
        sys.exit(1)

    configs = generate_configs(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        epochs=args.epochs,
    )

    if args.run:
        run_experiments(configs)
    else:
        print("\nConfigs generated. To run experiments:")
        for c in configs:
            print(f"  python src/training/train_landsat.py  # with config: {c['config_path']}")


if __name__ == "__main__":
    main()



