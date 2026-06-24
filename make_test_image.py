import numpy as np
from PIL import Image
from pathlib import Path

# Get first test sample
test_dir = Path('data/landsat9/splits/test')
sample_dir = next(test_dir.iterdir())

# Load TIR
tir = np.load(sample_dir / 'tir_100m.npy')

# Normalize to 0-255
tir = (tir - tir.min()) / (tir.max() - tir.min())
tir = (tir * 255).astype(np.uint8)

# Save
Image.fromarray(tir).save('test_tir_sample.png')
print(f"Saved: test_tir_sample.png")
print(f"From: {sample_dir.name}")