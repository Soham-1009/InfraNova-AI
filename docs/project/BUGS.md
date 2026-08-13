# InfraNova AI — Known Bugs and Quirks

This document lists known issues in the codebase that developers should be aware of. 

## 1. High Priority / Breaking

*No known high priority bugs at this time.*

## 2. Medium Priority / Misleading

*No known medium priority bugs at this time.*

## 3. Low Priority / Edge Cases

### 3.1 Dataset Pipeline Strict Shape Validation
- **Location**: `src/datasets/landsat9_dataset.py` (Line 250)
- **Description**: The dataset throws a hard error if any patch on disk does not perfectly match the configured `image_size`. It does not attempt to resize them dynamically.
- **Impact**: Changing `image_size` in config from 128 to 256 will crash the dataloader until the entire dataset is regenerated from Earth Engine.
- **Workaround**: If you change `image_size`, you must rerun the patch building scripts.
