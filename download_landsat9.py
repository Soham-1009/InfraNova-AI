"""
Download Landsat 9 TIR + RGB data via Google Drive export.
"""

import ee
import os
import time
from pathlib import Path

EE_PROJECT_ID = "infranova-ai"  # Your project ID

REGIONS = {
    # Original 5
    'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'name': 'Mumbai'},
    'delhi': {'lat': 28.6139, 'lon': 77.2090, 'name': 'Delhi'},
    'bangalore': {'lat': 12.9716, 'lon': 77.5946, 'name': 'Bangalore'},
    'chennai': {'lat': 13.0827, 'lon': 80.2707, 'name': 'Chennai'},
    'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'name': 'Kolkata'},
    'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'name': 'Hyderabad'},
    'ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'name': 'Ahmedabad'},
    'pune': {'lat': 18.5204, 'lon': 73.8567, 'name': 'Pune'},
    'jaipur': {'lat': 26.9124, 'lon': 75.7873, 'name': 'Jaipur'},
    'lucknow': {'lat': 26.8467, 'lon': 80.9462, 'name': 'Lucknow'},
    'kanpur': {'lat': 26.4499, 'lon': 80.3319, 'name': 'Kanpur'},
    'nagpur': {'lat': 21.1458, 'lon': 79.0882, 'name': 'Nagpur'},
    'indore': {'lat': 22.7196, 'lon': 75.8577, 'name': 'Indore'},
    'bhopal': {'lat': 23.2599, 'lon': 77.4126, 'name': 'Bhopal'},
    'patna': {'lat': 25.5941, 'lon': 85.1376, 'name': 'Patna'},
    'surat': {'lat': 21.1702, 'lon': 72.8311, 'name': 'Surat'},
    'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'name': 'Visakhapatnam'},
    'kochi': {'lat': 9.9312, 'lon': 76.2673, 'name': 'Kochi'},
    'guwahati': {'lat': 26.1445, 'lon': 91.7362, 'name': 'Guwahati'},
    'chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'name': 'Chandigarh'},
    'tokyo': {'lat': 35.6762, 'lon': 139.6503, 'name': 'Tokyo'},
    'newyork': {'lat': 40.7128, 'lon': -74.0060, 'name': 'New York'},
    'london': {'lat': 51.5074, 'lon': -0.1278, 'name': 'London'},
    'paris': {'lat': 48.8566, 'lon': 2.3522, 'name': 'Paris'},
    'sydney': {'lat': -33.8688, 'lon': 151.2093, 'name': 'Sydney'},
    'cairo': {'lat': 30.0444, 'lon': 31.2357, 'name': 'Cairo'},
    'rio': {'lat': -22.9068, 'lon': -43.1729, 'name': 'Rio'},
    'dubai': {'lat': 25.2048, 'lon': 55.2708, 'name': 'Dubai'},
    'singapore': {'lat': 1.3521, 'lon': 103.8198, 'name': 'Singapore'},
    'bangkok': {'lat': 13.7563, 'lon': 100.5018, 'name': 'Bangkok'},
    'moscow': {'lat': 55.7558, 'lon': 37.6173, 'name': 'Moscow'},
    'beijing': {'lat': 39.9042, 'lon': 116.4074, 'name': 'Beijing'},
    'seoul': {'lat': 37.5665, 'lon': 126.9780, 'name': 'Seoul'},
    'istanbul': {'lat': 41.0082, 'lon': 28.9784, 'name': 'Istanbul'},
    'capetown': {'lat': -33.9249, 'lon': 18.4241, 'name': 'Cape Town'},
    'lagos': {'lat': 6.5244, 'lon': 3.3792, 'name': 'Lagos'},
    'mexico': {'lat': 19.4326, 'lon': -99.1332, 'name': 'Mexico City'},
    'losangeles': {'lat': 34.0522, 'lon': -118.2437, 'name': 'Los Angeles'},
    'toronto': {'lat': 43.6532, 'lon': -79.3832, 'name': 'Toronto'},
    'buenosaires': {'lat': -34.6037, 'lon': -58.3816, 'name': 'Buenos Aires'},
    
    # Diverse landscapes for terrain variety
    'amazon': {'lat': -3.4653, 'lon': -62.2159, 'name': 'Amazon'},
    'sahara': {'lat': 23.4162, 'lon': 25.6628, 'name': 'Sahara'},
    'himalayas': {'lat': 27.9881, 'lon': 86.9250, 'name': 'Himalayas'},
    'ganges': {'lat': 25.3176, 'lon': 82.9739, 'name': 'Ganges'},
    'sundarbans': {'lat': 21.9497, 'lon': 89.1833, 'name': 'Sundarbans'},
    'andes': {'lat': -13.5320, 'lon': -71.9675, 'name': 'Andes'},
    'alps': {'lat': 46.5197, 'lon': 8.5500, 'name': 'Alps'},
    'congo': {'lat': -2.0, 'lon': 23.0, 'name': 'Congo Rainforest'},
    'siberia': {'lat': 60.0, 'lon': 100.0, 'name': 'Siberia'},
    'gobi': {'lat': 42.0, 'lon': 105.0, 'name': 'Gobi Desert'},
}

BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'ST_B10']
START_DATE = '2024-01-01'
END_DATE = '2024-06-30'

# Buffer in meters - smaller buffer for direct download compatibility
BUFFER_METERS = 15000  # 15km buffer (was 50km - too large)


def export_region(region_id, region_info):
    """Export Landsat 9 data for one region to Google Drive."""
    print(f"\nProcessing {region_info['name']}...")
    
    point = ee.Geometry.Point([region_info['lon'], region_info['lat']])
    region = point.buffer(BUFFER_METERS).bounds()
    
    collection = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filterBounds(point)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
        .select(BANDS)
    )
    
    count = collection.size().getInfo()
    print(f"  Found {count} images")
    
    if count == 0:
        return None
    
    image = collection.first().clip(region)
    
    # Export each band separately
    tasks = []
    for band in BANDS:
        task_name = f"{region_id}_{band}"
        task = ee.batch.Export.image.toDrive(
            image=image.select([band]),
            description=task_name,
            folder='InfraNova_Landsat9',
            fileNamePrefix=task_name,
            scale=30,
            region=region,
            maxPixels=1e9,
        )
        task.start()
        tasks.append((task_name, task))
        print(f"  Started export: {task_name}")
    
    return tasks


def monitor_tasks(all_tasks):
    """Monitor export tasks until complete."""
    print("\n" + "="*60)
    print("Monitoring export tasks...")
    print("This may take 5-15 minutes")
    print("="*60)
    
    while True:
        all_done = True
        for task_name, task in all_tasks:
            status = task.status()
            state = status['state']
            
            if state in ['READY', 'RUNNING']:
                all_done = False
                print(f"  {task_name}: {state}")
            elif state == 'COMPLETED':
                pass
            elif state == 'FAILED':
                print(f"  {task_name}: FAILED - {status.get('error_message', 'Unknown')}")
        
        if all_done:
            print("\nAll tasks completed!")
            break
        
        print("Waiting 30 seconds...")
        time.sleep(30)


def main():
    ee.Initialize(project=EE_PROJECT_ID)
    print("Earth Engine initialized")
    
    all_tasks = []
    
    for region_id, region_info in REGIONS.items():
        tasks = export_region(region_id, region_info)
        if tasks:
            all_tasks.extend(tasks)
    
    if not all_tasks:
        print("No tasks started")
        return
    
    print(f"\nStarted {len(all_tasks)} export tasks total")
    print("\nIMPORTANT: Files will be saved to your Google Drive")
    print("Folder: InfraNova_Landsat9")
    print("\nYou can monitor progress at: https://code.earthengine.google.com/tasks")
    
    monitor_tasks(all_tasks)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Go to: https://drive.google.com")
    print("2. Find folder: InfraNova_Landsat9")
    print("3. Download all files")
    print("4. Organize into data/landsat9/input/{region}_product/ folders")


if __name__ == '__main__':
    main()