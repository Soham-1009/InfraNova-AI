"""
Organize downloaded Landsat 9 files from Google Drive into proper folders.
"""

import shutil
from pathlib import Path

# Update this to where you extracted the Google Drive download
SOURCE_DIR = Path(r"C:\Users\soham\Downloads\InfraNova_Landsat9")

# Target folder structure
TARGET_DIR = Path("data/landsat9/input")

# All 20 regions
REGIONS = [
    # Original 20 Indian (already organized)
    'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata',
    'hyderabad', 'ahmedabad', 'pune', 'jaipur', 'lucknow',
    'kanpur', 'nagpur', 'indore', 'bhopal', 'patna',
    'surat', 'visakhapatnam', 'kochi', 'guwahati', 'chandigarh',
    
    # New international
    'tokyo', 'newyork', 'london', 'paris', 'sydney',
    'cairo', 'rio', 'dubai', 'singapore', 'bangkok',
    'moscow', 'beijing', 'seoul', 'istanbul', 'capetown',
    'lagos', 'mexico', 'losangeles', 'toronto', 'buenosaires',
    
    # Landscapes
    'amazon', 'sahara', 'himalayas', 'ganges', 'sundarbans',
    'andes', 'alps', 'congo', 'siberia', 'gobi',
]

def organize():
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    
    if not SOURCE_DIR.exists():
        print(f"\nERROR: Source folder not found!")
        print(f"Update SOURCE_DIR in this script to point to extracted folder")
        return
    
    moved_count = 0
    missing_files = []
    
    for region in REGIONS:
        region_target = TARGET_DIR / f"{region}_product"
        region_target.mkdir(parents=True, exist_ok=True)
        
        files_found = list(SOURCE_DIR.glob(f"{region}_*.tif"))
        
        if not files_found:
            print(f"WARNING: No files found for {region}")
            missing_files.append(region)
            continue
        
        for tif_file in files_found:
            target_path = region_target / tif_file.name
            
            if target_path.exists():
                print(f"  Skipping (exists): {tif_file.name}")
                continue
            
            shutil.move(str(tif_file), str(target_path))
            print(f"  Moved: {tif_file.name} -> {region}_product/")
            moved_count += 1
    
    print(f"\n{'='*60}")
    print(f"Total files moved: {moved_count}")
    
    if missing_files:
        print(f"\nMissing files for regions:")
        for region in missing_files:
            print(f"  - {region}")
    
    print(f"\nVerifying organized structure:")
    for region in REGIONS:
        region_dir = TARGET_DIR / f"{region}_product"
        if region_dir.exists():
            count = len(list(region_dir.glob("*.tif")))
            status = "OK" if count == 4 else f"INCOMPLETE ({count}/4)"
            print(f"  {region}: {status}")


if __name__ == '__main__':
    organize()