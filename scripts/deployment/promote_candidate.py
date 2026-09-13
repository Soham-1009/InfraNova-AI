"""
Production Promotion Script for InfraNova-AI.

Prepares and executes the promotion of candidate models to outputs/best/.
Default mode is DRY RUN. Requires explicit --execute flag.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.models.pix2pix.generator_hd import Pix2PixHDGlobalResNetGenerator


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Promote candidate model to production.")
    parser.add_argument(
        "--candidate",
        type=str,
        default="outputs/exp9/best/pix2pix_landsat_best.pth",
        help="Path to candidate checkpoint.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="outputs/best",
        help="Production target directory.",
    )
    parser.add_argument(
        "--expected-sha",
        type=str,
        default="71bbda3f31b85e7e741b26d5ce0ff398a0394c7dd4ef6ff5452f31f7e0400382",
        help="Expected SHA-256 for candidate.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the promotion. If not set, performs dry-run only.",
    )
    args = parser.parse_args()

    candidate_path = Path(args.candidate)
    target_dir = Path(args.target_dir)
    target_path = target_dir / "pix2pix_landsat_best.pth"

    print("=" * 60)
    print("INFRANOVA-AI PRODUCTION PROMOTION PROCEDURE")
    print("=" * 60)
    print(f"Candidate Checkpoint : {candidate_path}")
    print(f"Target Checkpoint    : {target_path}")
    print(f"Mode                 : {'EXECUTION' if args.execute else 'DRY-RUN (Safe Check Only)'}")
    print("-" * 60)

    # 1. Verify Candidate Existence & Checksum
    if not candidate_path.exists():
        print(f"ERROR: Candidate checkpoint does not exist: {candidate_path}")
        sys.exit(1)

    cand_sha = compute_sha256(candidate_path)
    print(f"[1/6] Candidate SHA-256: {cand_sha}")
    if args.expected_sha and cand_sha != args.expected_sha:
        print(f"ERROR: Candidate SHA-256 mismatch! Expected {args.expected_sha}, got {cand_sha}")
        sys.exit(1)
    print("      Candidate integrity verified.")

    # 2. Verify Candidate Architecture & Loadability
    print("[2/6] Validating candidate architecture & loadability...")
    try:
        ckpt = torch.load(candidate_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        gen_sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
        netG = Pix2PixHDGlobalResNetGenerator(in_channels=2, out_channels=3, ngf=64, n_blocks=9)
        netG.load_state_dict(gen_sd, strict=True)
        netG.eval()
        n_params = sum(p.numel() for p in netG.parameters())
        assert n_params == 11369795, f"Param count mismatch: {n_params}"
        print(f"      Architecture: Pix2PixHDGlobalResNetGenerator ({n_params:,} parameters). Verified.")
    except Exception as e:
        print(f"ERROR: Failed to load candidate checkpoint: {e}")
        sys.exit(1)

    # 3. Inference Validation Smoke Test
    print("[3/6] Validating dummy inference...")
    dummy_in = torch.randn(1, 2, 128, 128)
    with torch.no_grad():
        dummy_out = netG(dummy_in)
    assert dummy_out.shape == (1, 3, 128, 128)
    assert torch.isfinite(dummy_out).all().item()
    print("      Inference test passed (shape [1, 3, 128, 128], finite, 0 NaNs).")

    # 4. Check Current Production
    if target_path.exists():
        curr_sha = compute_sha256(target_path)
        print(f"[4/6] Current Production SHA-256: {curr_sha}")
    else:
        curr_sha = "NONE"
        print("[4/6] No current production file found.")

    if not args.execute:
        print("-" * 60)
        print("DRY-RUN COMPLETE.")
        print("To actually execute the promotion, run:")
        print("  python scripts/deployment/promote_candidate.py --execute")
        print("=" * 60)
        return

    # 5. Backup Current Production
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = target_dir / f"pix2pix_landsat_backup_{timestamp}.pth"
    rollback_json = target_dir / "rollback_metadata.json"

    if target_path.exists():
        print(f"[5/6] Backing up current production -> {backup_path}")
        shutil.copy2(target_path, backup_path)
        metadata = {
            "timestamp": timestamp,
            "previous_production_sha256": curr_sha,
            "backup_path": str(backup_path),
            "promoted_candidate": str(candidate_path),
            "promoted_candidate_sha256": cand_sha,
        }
        with open(rollback_json, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"      Rollback metadata written to {rollback_json}")

    # 6. Copy Candidate to Production
    print(f"[6/6] Promoting candidate to {target_path}...")
    shutil.copy2(candidate_path, target_path)
    new_prod_sha = compute_sha256(target_path)
    assert new_prod_sha == cand_sha, "Promoted file SHA mismatch!"
    print(f"      Promotion successful! New Production SHA-256: {new_prod_sha}")
    print("=" * 60)
    print("PROMOTION COMPLETED SUCCESSFULLY.")
    print("Rollback command if ever needed:")
    print(f"  cp '{backup_path}' '{target_path}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
