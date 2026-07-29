import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_stage(stage, command):
    print(f"\n{'='*50}\n🚀 Running stage: {stage.upper()}\n{'='*50}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"❌ Stage {stage} failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ Stage {stage} completed successfully.\n")

def main():
    parser = argparse.ArgumentParser(description="InfraNova AI End-to-End Pipeline Runner")
    parser.add_argument("--stage", type=str, choices=["download", "preprocess", "train", "evaluate", "all"], default="all")
    args = parser.parse_args()

    stages = {
        "download": [sys.executable, "scripts/download/download_landsat9.py"],
        "preprocess": [sys.executable, "scripts/preprocessing/split_patches.py"],
        "train": [sys.executable, "scripts/training/run_ablation_study.py"],
        "evaluate": [sys.executable, "scripts/evaluation/evaluate.py"]
    }

    if args.stage == "all":
        for s in stages:
            run_stage(s, stages[s])
    else:
        run_stage(args.stage, stages[args.stage])

if __name__ == "__main__":
    main()
