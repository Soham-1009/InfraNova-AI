"""
Generate Dataset Manifest

Creates a reproducible dataset_manifest.json with all parameters.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/soham/Desktop/Soham/InfraNova-AI")

def generate_manifest():
    out_path = PROJECT_ROOT / "data/landsat9/dataset_manifest.json"
    
    # Normally these would be pulled dynamically from constants, but we hardcode them here for the script
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "downloader_version": "1.1 (Adaptive Radius/Swath Edge)",
        "preprocessing_version": "2.0 (Direct GeoTIFF, Rich Stats)",
        "parameters": {
            "patch_size_200m": 64,
            "patch_size_100m": 128,
            "stride_200m": 16,
            "preprocess_scale": 0.15,
            "swath_edge_threshold": 0.10
        }
    }
    
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Dataset manifest saved to {out_path}")

if __name__ == "__main__":
    generate_manifest()
