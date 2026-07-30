"""
Orchestrator for Phase 1: Complete Dataset Generation & Verification
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path("c:/Users/soham/Desktop/Soham/InfraNova-AI")
PYTHON_EXE = sys.executable

def run_step(name: str, script: str, args: list = None) -> bool:
    print("\n" + "=" * 60)
    print(f"STEP: {name}")
    print("=" * 60)
    
    cmd = [PYTHON_EXE, str(PROJECT_ROOT / script)]
    if args:
        cmd.extend(args)
        
    print(f"Executing: {' '.join(cmd)}\n")
    
    try:
        # Stream output directly to the terminal
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step '{name}' failed with exit code {e.returncode}.")
        return False
    except Exception as e:
        print(f"\n[ERROR] Step '{name}' encountered an exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Dataset Pipeline Orchestrator")
    parser.add_argument("--cleanup", action="store_true", help="Automatically delete quarantine on success")
    args = parser.parse_args()
    
    steps = [
        ("Quarantine Undersized Regions", "scripts/preprocessing/quarantine_small.py", []),
        ("Download Missing/Replacement Regions", "scripts/download/download_landsat9.py", ["--workers", "8"]),
        ("Preprocess Patches", "scripts/preprocessing/process_landsat_patches.py", []),
        ("Automated Dataset Validation", "scripts/preprocessing/validate_patch_dataset.py", []),
        ("Generate Visual Inspection Grid", "scripts/preprocessing/visualize_grid.py", []),
        ("Generate Dataset Manifest", "scripts/preprocessing/generate_manifest.py", [])
    ]
    
    for name, script, script_args in steps:
        success = run_step(name, script, script_args)
        if not success:
            print("\n[FATAL] Pipeline halted due to step failure.")
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print("PIPELINE SUCCESS")
    print("All steps completed and validated successfully.")
    print("=" * 60)
    
    quarantine_dir = PROJECT_ROOT / "data/landsat9/quarantine"
    if quarantine_dir.exists() and any(quarantine_dir.iterdir()):
        if args.cleanup:
            print("\n--cleanup flag provided. Deleting quarantine directory...")
            shutil.rmtree(quarantine_dir)
            print("Quarantine directory removed.")
        else:
            print("\nQuarantine directory still exists with original folders.")
            print("If you are satisfied with the new dataset, you may safely delete:")
            print(f"  {quarantine_dir}")
            
    sys.exit(0)

if __name__ == "__main__":
    main()
