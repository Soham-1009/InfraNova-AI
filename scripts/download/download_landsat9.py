"""
Download Landsat 9 TIR + RGB data directly to local disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import io
import logging
import sys
import time
import zipfile
from contextlib import suppress
from pathlib import Path

import ee
import rasterio
import requests
import numpy as np

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EE_PROJECT_ID = "infranova-ai"  # Update if your project ID differs

REGIONS = {
    'delhi': {'lat': 28.6139, 'lon': 77.2090, 'name': 'Delhi'},
    'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'name': 'Mumbai'},
    'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'name': 'Kolkata'},
    'chennai': {'lat': 13.0827, 'lon': 80.2707, 'name': 'Chennai'},
    'bengaluru': {'lat': 12.9716, 'lon': 77.5946, 'name': 'Bengaluru'},
    'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'name': 'Hyderabad'},
    'pune': {'lat': 18.5204, 'lon': 73.8567, 'name': 'Pune'},
    'ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'name': 'Ahmedabad'},
    'jaipur': {'lat': 26.9124, 'lon': 75.7873, 'name': 'Jaipur'},
    'lucknow': {'lat': 26.8467, 'lon': 80.9462, 'name': 'Lucknow'},
    'kanpur': {'lat': 26.4499, 'lon': 80.3319, 'name': 'Kanpur'},
    'nagpur': {'lat': 21.1458, 'lon': 79.0882, 'name': 'Nagpur'},
    'indore': {'lat': 22.7196, 'lon': 75.8577, 'name': 'Indore'},
    'bhopal': {'lat': 23.2599, 'lon': 77.4126, 'name': 'Bhopal'},
    'patna': {'lat': 25.5941, 'lon': 85.1376, 'name': 'Patna'},
    'surat': {'lat': 21.1702, 'lon': 72.8311, 'name': 'Surat'},
    'vadodara': {'lat': 22.3072, 'lon': 73.1812, 'name': 'Vadodara'},
    'rajkot': {'lat': 22.3039, 'lon': 70.8022, 'name': 'Rajkot'},
    'nashik': {'lat': 19.9975, 'lon': 73.7898, 'name': 'Nashik'},
    'aurangabad': {'lat': 19.8762, 'lon': 75.3433, 'name': 'Aurangabad'},
    'solapur': {'lat': 17.6599, 'lon': 75.9064, 'name': 'Solapur'},
    'kolhapur': {'lat': 16.7050, 'lon': 74.2433, 'name': 'Kolhapur'},
    'amravati': {'lat': 20.9374, 'lon': 77.7796, 'name': 'Amravati'},
    'akola': {'lat': 20.7002, 'lon': 77.0082, 'name': 'Akola'},
    'nanded': {'lat': 19.1383, 'lon': 77.3210, 'name': 'Nanded'},
    'jalgaon': {'lat': 21.0077, 'lon': 75.5626, 'name': 'Jalgaon'},
    'sangli': {'lat': 16.8524, 'lon': 74.5815, 'name': 'Sangli'},
    'satara': {'lat': 17.6805, 'lon': 74.0183, 'name': 'Satara'},
    'ratnagiri': {'lat': 16.9902, 'lon': 73.3120, 'name': 'Ratnagiri'},
    'panaji': {'lat': 15.4909, 'lon': 73.8278, 'name': 'Panaji'},
    'mysuru': {'lat': 12.2958, 'lon': 76.6394, 'name': 'Mysuru'},
    'mangaluru': {'lat': 12.9141, 'lon': 74.8560, 'name': 'Mangaluru'},
    'hubballi': {'lat': 15.3647, 'lon': 75.1240, 'name': 'Hubballi'},
    'belagavi': {'lat': 15.8497, 'lon': 74.4977, 'name': 'Belagavi'},
    'shivamogga': {'lat': 13.9299, 'lon': 75.5681, 'name': 'Shivamogga'},
    'davanagere': {'lat': 14.4644, 'lon': 75.9218, 'name': 'Davanagere'},
    'ballari': {'lat': 15.1394, 'lon': 76.9214, 'name': 'Ballari'},
    'vijayapura': {'lat': 16.8302, 'lon': 75.7100, 'name': 'Vijayapura'},
    'kalaburagi': {'lat': 17.3297, 'lon': 76.8343, 'name': 'Kalaburagi'},
    'udupi': {'lat': 13.3409, 'lon': 74.7421, 'name': 'Udupi'},
    'kochi': {'lat': 9.9312, 'lon': 76.2673, 'name': 'Kochi'},
    'thiruvananthapuram': {'lat': 8.5241, 'lon': 76.9366, 'name': 'Thiruvananthapuram'},
    'kozhikode': {'lat': 11.2588, 'lon': 75.7804, 'name': 'Kozhikode'},
    'thrissur': {'lat': 10.5276, 'lon': 76.2144, 'name': 'Thrissur'},
    'kannur': {'lat': 11.8745, 'lon': 75.3704, 'name': 'Kannur'},
    'alappuzha': {'lat': 9.4981, 'lon': 76.3388, 'name': 'Alappuzha'},
    'kollam': {'lat': 8.8932, 'lon': 76.6141, 'name': 'Kollam'},
    'palakkad': {'lat': 10.7867, 'lon': 76.6548, 'name': 'Palakkad'},
    'kottayam': {'lat': 9.5916, 'lon': 76.5222, 'name': 'Kottayam'},
    'malappuram': {'lat': 11.0510, 'lon': 76.0711, 'name': 'Malappuram'},
    'coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'name': 'Coimbatore'},
    'madurai': {'lat': 9.9252, 'lon': 78.1198, 'name': 'Madurai'},
    'tiruchirappalli': {'lat': 10.7905, 'lon': 78.7047, 'name': 'Tiruchirappalli'},
    'salem': {'lat': 11.6643, 'lon': 78.1460, 'name': 'Salem'},
    'tirunelveli': {'lat': 8.7139, 'lon': 77.7567, 'name': 'Tirunelveli'},
    'erode': {'lat': 11.3410, 'lon': 77.7172, 'name': 'Erode'},
    'vellore': {'lat': 12.9165, 'lon': 79.1325, 'name': 'Vellore'},
    'thoothukudi': {'lat': 8.7642, 'lon': 78.1348, 'name': 'Thoothukudi'},
    'dindigul': {'lat': 10.3673, 'lon': 77.9803, 'name': 'Dindigul'},
    'thanjavur': {'lat': 10.7867, 'lon': 79.1378, 'name': 'Thanjavur'},
    'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'name': 'Visakhapatnam'},
    'vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'name': 'Vijayawada'},
    'guntur': {'lat': 16.3067, 'lon': 80.4365, 'name': 'Guntur'},
    'kurnool': {'lat': 15.8281, 'lon': 78.0373, 'name': 'Kurnool'},
    'rajahmundry': {'lat': 17.0005, 'lon': 81.8040, 'name': 'Rajahmundry'},
    'nellore': {'lat': 14.4426, 'lon': 79.9865, 'name': 'Nellore'},
    'tirupati': {'lat': 13.6288, 'lon': 79.4192, 'name': 'Tirupati'},
    'kadapa': {'lat': 14.4673, 'lon': 78.8242, 'name': 'Kadapa'},
    'anantapur': {'lat': 14.6819, 'lon': 77.6006, 'name': 'Anantapur'},
    'kakinada': {'lat': 16.9891, 'lon': 82.2475, 'name': 'Kakinada'},
    'warangal': {'lat': 17.9689, 'lon': 79.5941, 'name': 'Warangal'},
    'nizamabad': {'lat': 18.6725, 'lon': 78.0941, 'name': 'Nizamabad'},
    'karimnagar': {'lat': 18.4386, 'lon': 79.1288, 'name': 'Karimnagar'},
    'khammam': {'lat': 17.2473, 'lon': 80.1514, 'name': 'Khammam'},
    'ramagundam': {'lat': 18.7611, 'lon': 79.4741, 'name': 'Ramagundam'},
    'mahbubnagar': {'lat': 16.7488, 'lon': 77.9851, 'name': 'Mahbubnagar'},
    'suryapet': {'lat': 17.1405, 'lon': 79.6207, 'name': 'Suryapet'},
    'adilabad': {'lat': 19.6667, 'lon': 78.5333, 'name': 'Adilabad'},
    'siddipet': {'lat': 18.1048, 'lon': 78.8486, 'name': 'Siddipet'},
    'jagtial': {'lat': 18.7907, 'lon': 78.9120, 'name': 'Jagtial'},
    'bhubaneswar': {'lat': 20.2961, 'lon': 85.8245, 'name': 'Bhubaneswar'},
    'cuttack': {'lat': 20.4625, 'lon': 85.8830, 'name': 'Cuttack'},
    'rourkela': {'lat': 22.2604, 'lon': 84.8536, 'name': 'Rourkela'},
    'sambalpur': {'lat': 21.4704, 'lon': 83.9701, 'name': 'Sambalpur'},
    'berhampur': {'lat': 19.3149, 'lon': 84.7941, 'name': 'Berhampur'},
    'raipur': {'lat': 21.2514, 'lon': 81.6296, 'name': 'Raipur'},
    'bhilai': {'lat': 21.1938, 'lon': 81.3509, 'name': 'Bhilai'},
    'bilaspur': {'lat': 22.0796, 'lon': 82.1409, 'name': 'Bilaspur'},
    'korba': {'lat': 22.3595, 'lon': 82.7501, 'name': 'Korba'},
    'jagdalpur': {'lat': 19.0741, 'lon': 82.0080, 'name': 'Jagdalpur'},
    'ranchi': {'lat': 23.3441, 'lon': 85.3096, 'name': 'Ranchi'},
    'jamshedpur': {'lat': 22.8046, 'lon': 86.2029, 'name': 'Jamshedpur'},
    'dhanbad': {'lat': 23.7957, 'lon': 86.4304, 'name': 'Dhanbad'},
    'bokaro': {'lat': 23.6693, 'lon': 86.1511, 'name': 'Bokaro'},
    'hazaribagh': {'lat': 23.9966, 'lon': 85.3691, 'name': 'Hazaribagh'},
    'chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'name': 'Chandigarh'},
    'ludhiana': {'lat': 30.9010, 'lon': 75.8573, 'name': 'Ludhiana'},
    'amritsar': {'lat': 31.6340, 'lon': 74.8723, 'name': 'Amritsar'},
    'jalandhar': {'lat': 31.3260, 'lon': 75.5762, 'name': 'Jalandhar'},
    'patiala': {'lat': 30.3398, 'lon': 76.3869, 'name': 'Patiala'},
    'shimla': {'lat': 31.1048, 'lon': 77.1734, 'name': 'Shimla'},
    'dharamshala': {'lat': 32.2190, 'lon': 76.3234, 'name': 'Dharamshala'},
    'mandi': {'lat': 31.7084, 'lon': 76.9314, 'name': 'Mandi'},
    'solan': {'lat': 30.9045, 'lon': 77.0967, 'name': 'Solan'},
    'hamirpur_hp': {'lat': 31.6840, 'lon': 76.5255, 'name': 'Hamirpur'},
    'una': {'lat': 31.4685, 'lon': 76.2708, 'name': 'Una'},
    'chamba': {'lat': 32.5530, 'lon': 76.1258, 'name': 'Chamba'},
    'kullu': {'lat': 31.9579, 'lon': 77.1095, 'name': 'Kullu'},
    'manali': {'lat': 32.2432, 'lon': 77.1892, 'name': 'Manali'},
    'bilaspur_hp': {'lat': 31.3315, 'lon': 76.7566, 'name': 'Bilaspur'},
    'dehradun': {'lat': 30.3165, 'lon': 78.0322, 'name': 'Dehradun'},
    'haridwar': {'lat': 29.9457, 'lon': 78.1642, 'name': 'Haridwar'},
    'rishikesh': {'lat': 30.0869, 'lon': 78.2676, 'name': 'Rishikesh'},
    'haldwani': {'lat': 29.2183, 'lon': 79.5120, 'name': 'Haldwani'},
    'nainital': {'lat': 29.3919, 'lon': 79.4542, 'name': 'Nainital'},
    'rudrapur': {'lat': 28.9875, 'lon': 79.4141, 'name': 'Rudrapur'},
    'almora': {'lat': 29.5971, 'lon': 79.6591, 'name': 'Almora'},
    'pithoragarh': {'lat': 29.5829, 'lon': 80.2182, 'name': 'Pithoragarh'},
    'srinagar_uk': {'lat': 30.2224, 'lon': 78.7834, 'name': 'Srinagar (Uttarakhand)'},
    'uttarkashi': {'lat': 30.7290, 'lon': 78.4430, 'name': 'Uttarkashi'},
    'srinagar': {'lat': 34.0837, 'lon': 74.7973, 'name': 'Srinagar'},
    'jammu': {'lat': 32.7266, 'lon': 74.8570, 'name': 'Jammu'},
    'anantnag': {'lat': 33.7311, 'lon': 75.1542, 'name': 'Anantnag'},
    'baramulla': {'lat': 34.1980, 'lon': 74.3636, 'name': 'Baramulla'},
    'pulwama': {'lat': 33.8740, 'lon': 74.8996, 'name': 'Pulwama'},
    'kupwara': {'lat': 34.5260, 'lon': 74.2642, 'name': 'Kupwara'},
    'udhampur': {'lat': 32.9253, 'lon': 75.1352, 'name': 'Udhampur'},
    'kargil': {'lat': 34.5539, 'lon': 76.1349, 'name': 'Kargil'},
    'leh': {'lat': 34.1526, 'lon': 77.5771, 'name': 'Leh'},
    'drass': {'lat': 34.4300, 'lon': 75.7570, 'name': 'Drass'},
    'guwahati': {'lat': 26.1445, 'lon': 91.7362, 'name': 'Guwahati'},
    'dibrugarh': {'lat': 27.4728, 'lon': 94.9120, 'name': 'Dibrugarh'},
    'silchar': {'lat': 24.8333, 'lon': 92.7789, 'name': 'Silchar'},
    'jorhat': {'lat': 26.7509, 'lon': 94.2037, 'name': 'Jorhat'},
    'tezpur': {'lat': 26.6528, 'lon': 92.7926, 'name': 'Tezpur'},
    'nagaon': {'lat': 26.3480, 'lon': 92.6840, 'name': 'Nagaon'},
    'tinsukia': {'lat': 27.4891, 'lon': 95.3599, 'name': 'Tinsukia'},
    'dimapur': {'lat': 25.9091, 'lon': 93.7266, 'name': 'Dimapur'},
    'kohima': {'lat': 25.6751, 'lon': 94.1086, 'name': 'Kohima'},
    'mokokchung': {'lat': 26.3248, 'lon': 94.5183, 'name': 'Mokokchung'},
    'itanagar': {'lat': 27.0844, 'lon': 93.6053, 'name': 'Itanagar'},
    'pasighat': {'lat': 28.0661, 'lon': 95.3268, 'name': 'Pasighat'},
    'tawang': {'lat': 27.5861, 'lon': 91.8594, 'name': 'Tawang'},
    'ziro': {'lat': 27.5454, 'lon': 93.8197, 'name': 'Ziro'},
    'bomdila': {'lat': 27.2648, 'lon': 92.4241, 'name': 'Bomdila'},
    'shillong': {'lat': 25.5788, 'lon': 91.8933, 'name': 'Shillong'},
    'tura': {'lat': 25.5142, 'lon': 90.2021, 'name': 'Tura'},
    'jowai': {'lat': 25.4520, 'lon': 92.2060, 'name': 'Jowai'},
    'nongstoin': {'lat': 25.5170, 'lon': 91.2640, 'name': 'Nongstoin'},
    'baghmara': {'lat': 25.2050, 'lon': 90.6460, 'name': 'Baghmara'},
    'imphal': {'lat': 24.8170, 'lon': 93.9368, 'name': 'Imphal'},
    'churachandpur': {'lat': 24.3335, 'lon': 93.6766, 'name': 'Churachandpur'},
    'ukhrul': {'lat': 25.1216, 'lon': 94.3565, 'name': 'Ukhrul'},
    'thoubal': {'lat': 24.6385, 'lon': 93.9964, 'name': 'Thoubal'},
    'bishnupur_mn': {'lat': 24.6270, 'lon': 93.7660, 'name': 'Bishnupur'},
    'aizawl': {'lat': 23.7271, 'lon': 92.7176, 'name': 'Aizawl'},
    'lunglei': {'lat': 22.8870, 'lon': 92.7340, 'name': 'Lunglei'},
    'kolasib': {'lat': 24.2230, 'lon': 92.6780, 'name': 'Kolasib'},
    'champhai': {'lat': 23.4560, 'lon': 93.3280, 'name': 'Champhai'},
    'serchhip': {'lat': 23.2980, 'lon': 92.8460, 'name': 'Serchhip'},
    'agartala': {'lat': 23.8315, 'lon': 91.2868, 'name': 'Agartala'},
    'udaipur_tr': {'lat': 23.5330, 'lon': 91.4830, 'name': 'Udaipur'},
    'dharmanagar': {'lat': 24.3660, 'lon': 92.1660, 'name': 'Dharmanagar'},
    'kailashahar': {'lat': 24.3320, 'lon': 92.0040, 'name': 'Kailashahar'},
    'ambassa': {'lat': 23.9360, 'lon': 91.8540, 'name': 'Ambassa'},
    'gangtok': {'lat': 27.3389, 'lon': 88.6065, 'name': 'Gangtok'},
    'namchi': {'lat': 27.1660, 'lon': 88.3630, 'name': 'Namchi'},
    'gyalshing': {'lat': 27.2890, 'lon': 88.2570, 'name': 'Gyalshing'},
    'mangan': {'lat': 27.5170, 'lon': 88.5340, 'name': 'Mangan'},
    'gurugram': {'lat': 28.4595, 'lon': 77.0266, 'name': 'Gurugram'},
    'faridabad': {'lat': 28.4089, 'lon': 77.3178, 'name': 'Faridabad'},
    'panipat': {'lat': 29.3909, 'lon': 76.9635, 'name': 'Panipat'},
    'ambala': {'lat': 30.3752, 'lon': 76.7821, 'name': 'Ambala'},
    'hisar': {'lat': 29.1492, 'lon': 75.7217, 'name': 'Hisar'},
    'meerut': {'lat': 28.9845, 'lon': 77.7064, 'name': 'Meerut'},
    'agra': {'lat': 27.1767, 'lon': 78.0081, 'name': 'Agra'},
    'varanasi': {'lat': 25.3176, 'lon': 82.9739, 'name': 'Varanasi'},
    'prayagraj': {'lat': 25.4358, 'lon': 81.8463, 'name': 'Prayagraj'},
    'gorakhpur': {'lat': 26.7606, 'lon': 83.3732, 'name': 'Gorakhpur'},
    'gaya': {'lat': 24.7914, 'lon': 85.0002, 'name': 'Gaya'},
    'muzaffarpur': {'lat': 26.1209, 'lon': 85.3647, 'name': 'Muzaffarpur'},
    'darbhanga': {'lat': 26.1542, 'lon': 85.8918, 'name': 'Darbhanga'},
    'bhagalpur': {'lat': 25.2425, 'lon': 86.9842, 'name': 'Bhagalpur'},
    'purnia': {'lat': 25.7781, 'lon': 87.4753, 'name': 'Purnia'},
    'udaipur': {'lat': 24.5854, 'lon': 73.7125, 'name': 'Udaipur'},
    'jodhpur': {'lat': 26.2389, 'lon': 73.0243, 'name': 'Jodhpur'},
    'kota': {'lat': 25.2138, 'lon': 75.8648, 'name': 'Kota'},
    'ajmer': {'lat': 26.4499, 'lon': 74.6399, 'name': 'Ajmer'},
    'bikaner': {'lat': 28.0229, 'lon': 73.3119, 'name': 'Bikaner'},
    'gwalior': {'lat': 26.2183, 'lon': 78.1828, 'name': 'Gwalior'},
    'jabalpur': {'lat': 23.1815, 'lon': 79.9864, 'name': 'Jabalpur'},
    'sagar': {'lat': 23.8388, 'lon': 78.7378, 'name': 'Sagar'},
    'rewa': {'lat': 24.5362, 'lon': 81.3037, 'name': 'Rewa'},
    'ujjain': {'lat': 23.1765, 'lon': 75.7885, 'name': 'Ujjain'},
    'dewas': {'lat': 22.9676, 'lon': 76.0534, 'name': 'Dewas'},
    'satna': {'lat': 24.5773, 'lon': 80.8272, 'name': 'Satna'},
    'chhindwara': {'lat': 22.0574, 'lon': 78.9382, 'name': 'Chhindwara'},
    'burhanpur': {'lat': 21.3074, 'lon': 76.2303, 'name': 'Burhanpur'},
    'katni': {'lat': 23.8388, 'lon': 80.3940, 'name': 'Katni'},
    'singrauli': {'lat': 24.1998, 'lon': 82.6753, 'name': 'Singrauli'},

    # 200 international cities
    'new_york': {'lat': 40.7128, 'lon': -74.0060, 'name': 'New York'},
    'los_angeles': {'lat': 34.0522, 'lon': -118.2437, 'name': 'Los Angeles'},
    'chicago': {'lat': 41.8781, 'lon': -87.6298, 'name': 'Chicago'},
    'houston': {'lat': 29.7604, 'lon': -95.3698, 'name': 'Houston'},
    'phoenix': {'lat': 33.4484, 'lon': -112.0740, 'name': 'Phoenix'},
    'philadelphia': {'lat': 39.9526, 'lon': -75.1652, 'name': 'Philadelphia'},
    'san_antonio': {'lat': 29.4241, 'lon': -98.4936, 'name': 'San Antonio'},
    'san_diego': {'lat': 32.7157, 'lon': -117.1611, 'name': 'San Diego'},
    'dallas': {'lat': 32.7767, 'lon': -96.7970, 'name': 'Dallas'},
    'san_jose': {'lat': 37.3382, 'lon': -121.8863, 'name': 'San Jose'},
    'austin': {'lat': 30.2672, 'lon': -97.7431, 'name': 'Austin'},
    'jacksonville': {'lat': 30.3322, 'lon': -81.6557, 'name': 'Jacksonville'},
    'fort_worth': {'lat': 32.7555, 'lon': -97.3308, 'name': 'Fort Worth'},
    'columbus': {'lat': 39.9612, 'lon': -82.9988, 'name': 'Columbus'},
    'charlotte': {'lat': 35.2271, 'lon': -80.8431, 'name': 'Charlotte'},
    'san_francisco': {'lat': 37.7749, 'lon': -122.4194, 'name': 'San Francisco'},
    'indianapolis': {'lat': 39.7684, 'lon': -86.1581, 'name': 'Indianapolis'},
    'seattle': {'lat': 47.6062, 'lon': -122.3321, 'name': 'Seattle'},
    'denver': {'lat': 39.7392, 'lon': -104.9903, 'name': 'Denver'},
    'washington_dc': {'lat': 38.9072, 'lon': -77.0369, 'name': 'Washington DC'},
    'boston': {'lat': 42.3601, 'lon': -71.0589, 'name': 'Boston'},
    'detroit': {'lat': 42.3314, 'lon': -83.0458, 'name': 'Detroit'},
    'nashville': {'lat': 36.1627, 'lon': -86.7816, 'name': 'Nashville'},
    'portland': {'lat': 45.5152, 'lon': -122.6784, 'name': 'Portland'},
    'las_vegas': {'lat': 36.1699, 'lon': -115.1398, 'name': 'Las Vegas'},
    'miami': {'lat': 25.7617, 'lon': -80.1918, 'name': 'Miami'},
    'minneapolis': {'lat': 44.9778, 'lon': -93.2650, 'name': 'Minneapolis'},
    'new_orleans': {'lat': 29.9511, 'lon': -90.0715, 'name': 'New Orleans'},
    'salt_lake_city': {'lat': 40.7608, 'lon': -111.8910, 'name': 'Salt Lake City'},
    'kansas_city': {'lat': 39.0997, 'lon': -94.5786, 'name': 'Kansas City'},
    'toronto': {'lat': 43.6532, 'lon': -79.3832, 'name': 'Toronto'},
    'vancouver': {'lat': 49.2827, 'lon': -123.1207, 'name': 'Vancouver'},
    'montreal': {'lat': 45.5019, 'lon': -73.5674, 'name': 'Montreal'},
    'calgary': {'lat': 51.0447, 'lon': -114.0719, 'name': 'Calgary'},
    'ottawa': {'lat': 45.4215, 'lon': -75.6972, 'name': 'Ottawa'},
    'edmonton': {'lat': 53.5461, 'lon': -113.4938, 'name': 'Edmonton'},
    'quebec_city': {'lat': 46.8139, 'lon': -71.2080, 'name': 'Quebec City'},
    'winnipeg': {'lat': 49.8951, 'lon': -97.1384, 'name': 'Winnipeg'},
    'halifax': {'lat': 44.6488, 'lon': -63.5752, 'name': 'Halifax'},
    'victoria': {'lat': 48.4284, 'lon': -123.3656, 'name': 'Victoria'},
    'mexico_city': {'lat': 19.4326, 'lon': -99.1332, 'name': 'Mexico City'},
    'guadalajara': {'lat': 20.6597, 'lon': -103.3496, 'name': 'Guadalajara'},
    'monterrey': {'lat': 25.6866, 'lon': -100.3161, 'name': 'Monterrey'},
    'puebla': {'lat': 19.0414, 'lon': -98.2063, 'name': 'Puebla'},
    'tijuana': {'lat': 32.5149, 'lon': -117.0382, 'name': 'Tijuana'},
    'merida': {'lat': 20.9674, 'lon': -89.5926, 'name': 'Merida'},
    'leon_mexico': {'lat': 21.1220, 'lon': -101.6840, 'name': 'León'},
    'queretaro': {'lat': 20.5888, 'lon': -100.3899, 'name': 'Querétaro'},
    'cancun': {'lat': 21.1619, 'lon': -86.8515, 'name': 'Cancún'},
    'chihuahua': {'lat': 28.6329, 'lon': -106.0691, 'name': 'Chihuahua'},
    'london': {'lat': 51.5074, 'lon': -0.1278, 'name': 'London'},
    'birmingham': {'lat': 52.4862, 'lon': -1.8904, 'name': 'Birmingham'},
    'manchester': {'lat': 53.4808, 'lon': -2.2426, 'name': 'Manchester'},
    'glasgow': {'lat': 55.8642, 'lon': -4.2518, 'name': 'Glasgow'},
    'edinburgh': {'lat': 55.9533, 'lon': -3.1883, 'name': 'Edinburgh'},
    'liverpool': {'lat': 53.4084, 'lon': -2.9916, 'name': 'Liverpool'},
    'bristol': {'lat': 51.4545, 'lon': -2.5879, 'name': 'Bristol'},
    'leeds': {'lat': 53.8008, 'lon': -1.5491, 'name': 'Leeds'},
    'cardiff': {'lat': 51.4816, 'lon': -3.1791, 'name': 'Cardiff'},
    'belfast': {'lat': 54.5973, 'lon': -5.9301, 'name': 'Belfast'},
    'paris': {'lat': 48.8566, 'lon': 2.3522, 'name': 'Paris'},
    'lyon': {'lat': 45.7640, 'lon': 4.8357, 'name': 'Lyon'},
    'marseille': {'lat': 43.2965, 'lon': 5.3698, 'name': 'Marseille'},
    'toulouse': {'lat': 43.6047, 'lon': 1.4442, 'name': 'Toulouse'},
    'nice': {'lat': 43.7102, 'lon': 7.2620, 'name': 'Nice'},
    'berlin': {'lat': 52.5200, 'lon': 13.4050, 'name': 'Berlin'},
    'hamburg': {'lat': 53.5511, 'lon': 9.9937, 'name': 'Hamburg'},
    'munich': {'lat': 48.1351, 'lon': 11.5820, 'name': 'Munich'},
    'frankfurt': {'lat': 50.1109, 'lon': 8.6821, 'name': 'Frankfurt'},
    'cologne': {'lat': 50.9375, 'lon': 6.9603, 'name': 'Cologne'},
    'rome': {'lat': 41.9028, 'lon': 12.4964, 'name': 'Rome'},
    'milan': {'lat': 45.4642, 'lon': 9.1900, 'name': 'Milan'},
    'naples': {'lat': 40.8518, 'lon': 14.2681, 'name': 'Naples'},
    'turin': {'lat': 45.0703, 'lon': 7.6869, 'name': 'Turin'},
    'florence': {'lat': 43.7696, 'lon': 11.2558, 'name': 'Florence'},
    'madrid': {'lat': 40.4168, 'lon': -3.7038, 'name': 'Madrid'},
    'barcelona': {'lat': 41.3851, 'lon': 2.1734, 'name': 'Barcelona'},
    'valencia': {'lat': 39.4699, 'lon': -0.3763, 'name': 'Valencia'},
    'seville': {'lat': 37.3891, 'lon': -5.9845, 'name': 'Seville'},
    'bilbao': {'lat': 43.2630, 'lon': -2.9350, 'name': 'Bilbao'},
    'amsterdam': {'lat': 52.3676, 'lon': 4.9041, 'name': 'Amsterdam'},
    'rotterdam': {'lat': 51.9244, 'lon': 4.4777, 'name': 'Rotterdam'},
    'brussels': {'lat': 50.8503, 'lon': 4.3517, 'name': 'Brussels'},
    'antwerp': {'lat': 51.2194, 'lon': 4.4025, 'name': 'Antwerp'},
    'luxembourg_city': {'lat': 49.6116, 'lon': 6.1319, 'name': 'Luxembourg City'},
    'zurich': {'lat': 47.3769, 'lon': 8.5417, 'name': 'Zurich'},
    'geneva': {'lat': 46.2044, 'lon': 6.1432, 'name': 'Geneva'},
    'vienna': {'lat': 48.2082, 'lon': 16.3738, 'name': 'Vienna'},
    'salzburg': {'lat': 47.8095, 'lon': 13.0550, 'name': 'Salzburg'},
    'prague': {'lat': 50.0755, 'lon': 14.4378, 'name': 'Prague'},
    'warsaw': {'lat': 52.2297, 'lon': 21.0122, 'name': 'Warsaw'},
    'krakow': {'lat': 50.0647, 'lon': 19.9450, 'name': 'Kraków'},
    'budapest': {'lat': 47.4979, 'lon': 19.0402, 'name': 'Budapest'},
    'athens': {'lat': 37.9838, 'lon': 23.7275, 'name': 'Athens'},
    'lisbon': {'lat': 38.7223, 'lon': -9.1393, 'name': 'Lisbon'},
    'dublin': {'lat': 53.3498, 'lon': -6.2603, 'name': 'Dublin'},
    'oslo': {'lat': 59.9139, 'lon': 10.7522, 'name': 'Oslo'},
    'stockholm': {'lat': 59.3293, 'lon': 18.0686, 'name': 'Stockholm'},
    'copenhagen': {'lat': 55.6761, 'lon': 12.5683, 'name': 'Copenhagen'},
    'helsinki': {'lat': 60.1699, 'lon': 24.9384, 'name': 'Helsinki'},
    'tokyo': {'lat': 35.6762, 'lon': 139.6503, 'name': 'Tokyo'},
    'osaka': {'lat': 34.6937, 'lon': 135.5023, 'name': 'Osaka'},
    'kyoto': {'lat': 35.0116, 'lon': 135.7681, 'name': 'Kyoto'},
    'yokohama': {'lat': 35.4437, 'lon': 139.6380, 'name': 'Yokohama'},
    'nagoya': {'lat': 35.1815, 'lon': 136.9066, 'name': 'Nagoya'},
    'sapporo': {'lat': 43.0618, 'lon': 141.3545, 'name': 'Sapporo'},
    'fukuoka': {'lat': 33.5902, 'lon': 130.4017, 'name': 'Fukuoka'},
    'hiroshima': {'lat': 34.3853, 'lon': 132.4553, 'name': 'Hiroshima'},
    'sendai': {'lat': 38.2682, 'lon': 140.8694, 'name': 'Sendai'},
    'naha': {'lat': 26.2124, 'lon': 127.6809, 'name': 'Naha'},
    'seoul': {'lat': 37.5665, 'lon': 126.9780, 'name': 'Seoul'},
    'busan': {'lat': 35.1796, 'lon': 129.0756, 'name': 'Busan'},
    'incheon': {'lat': 37.4563, 'lon': 126.7052, 'name': 'Incheon'},
    'daegu': {'lat': 35.8714, 'lon': 128.6014, 'name': 'Daegu'},
    'daejeon': {'lat': 36.3504, 'lon': 127.3845, 'name': 'Daejeon'},
    'beijing': {'lat': 39.9042, 'lon': 116.4074, 'name': 'Beijing'},
    'shanghai': {'lat': 31.2304, 'lon': 121.4737, 'name': 'Shanghai'},
    'guangzhou': {'lat': 23.1291, 'lon': 113.2644, 'name': 'Guangzhou'},
    'shenzhen': {'lat': 22.5431, 'lon': 114.0579, 'name': 'Shenzhen'},
    'chengdu': {'lat': 30.5728, 'lon': 104.0668, 'name': 'Chengdu'},
    'chongqing': {'lat': 29.5630, 'lon': 106.5516, 'name': 'Chongqing'},
    'wuhan': {'lat': 30.5928, 'lon': 114.3055, 'name': 'Wuhan'},
    'xian': {'lat': 34.3416, 'lon': 108.9398, 'name': "Xi'an"},
    'hangzhou': {'lat': 30.2741, 'lon': 120.1551, 'name': 'Hangzhou'},
    'nanjing': {'lat': 32.0603, 'lon': 118.7969, 'name': 'Nanjing'},
    'hong_kong': {'lat': 22.3193, 'lon': 114.1694, 'name': 'Hong Kong'},
    'taipei': {'lat': 25.0330, 'lon': 121.5654, 'name': 'Taipei'},
    'kaohsiung': {'lat': 22.6273, 'lon': 120.3014, 'name': 'Kaohsiung'},
    'singapore': {'lat': 1.3521, 'lon': 103.8198, 'name': 'Singapore'},
    'kuala_lumpur': {'lat': 3.1390, 'lon': 101.6869, 'name': 'Kuala Lumpur'},
    'george_town': {'lat': 5.4141, 'lon': 100.3288, 'name': 'George Town'},
    'johor_bahru': {'lat': 1.4927, 'lon': 103.7414, 'name': 'Johor Bahru'},
    'bangkok': {'lat': 13.7563, 'lon': 100.5018, 'name': 'Bangkok'},
    'chiang_mai': {'lat': 18.7883, 'lon': 98.9853, 'name': 'Chiang Mai'},
    'phuket': {'lat': 7.8804, 'lon': 98.3923, 'name': 'Phuket'},
    'hanoi': {'lat': 21.0278, 'lon': 105.8342, 'name': 'Hanoi'},
    'ho_chi_minh_city': {'lat': 10.8231, 'lon': 106.6297, 'name': 'Ho Chi Minh City'},
    'da_nang': {'lat': 16.0544, 'lon': 108.2022, 'name': 'Da Nang'},
    'jakarta': {'lat': -6.2088, 'lon': 106.8456, 'name': 'Jakarta'},
    'surabaya': {'lat': -7.2575, 'lon': 112.7521, 'name': 'Surabaya'},
    'bandung': {'lat': -6.9175, 'lon': 107.6191, 'name': 'Bandung'},
    'manila': {'lat': 14.5995, 'lon': 120.9842, 'name': 'Manila'},
    'cebu': {'lat': 10.3157, 'lon': 123.8854, 'name': 'Cebu'},
    'davao': {'lat': 7.1907, 'lon': 125.4553, 'name': 'Davao'},
    'dubai': {'lat': 25.2048, 'lon': 55.2708, 'name': 'Dubai'},
    'abu_dhabi': {'lat': 24.4539, 'lon': 54.3773, 'name': 'Abu Dhabi'},
    'riyadh': {'lat': 24.7136, 'lon': 46.6753, 'name': 'Riyadh'},
    'jeddah': {'lat': 21.4858, 'lon': 39.1925, 'name': 'Jeddah'},
    'doha': {'lat': 25.2854, 'lon': 51.5310, 'name': 'Doha'},
    'kuwait_city': {'lat': 29.3759, 'lon': 47.9774, 'name': 'Kuwait City'},
    'muscat': {'lat': 23.5880, 'lon': 58.3829, 'name': 'Muscat'},
    'amman': {'lat': 31.9454, 'lon': 35.9284, 'name': 'Amman'},
    'tel_aviv': {'lat': 32.0853, 'lon': 34.7818, 'name': 'Tel Aviv'},
    'jerusalem': {'lat': 31.7683, 'lon': 35.2137, 'name': 'Jerusalem'},
    'sao_paulo': {'lat': -23.5505, 'lon': -46.6333, 'name': 'São Paulo'},
    'rio_de_janeiro': {'lat': -22.9068, 'lon': -43.1729, 'name': 'Rio de Janeiro'},
    'brasilia': {'lat': -15.7939, 'lon': -47.8828, 'name': 'Brasília'},
    'belo_horizonte': {'lat': -19.9167, 'lon': -43.9345, 'name': 'Belo Horizonte'},
    'porto_alegre': {'lat': -30.0346, 'lon': -51.2177, 'name': 'Porto Alegre'},
    'curitiba': {'lat': -25.4284, 'lon': -49.2733, 'name': 'Curitiba'},
    'salvador': {'lat': -12.9777, 'lon': -38.5016, 'name': 'Salvador'},
    'fortaleza': {'lat': -3.7319, 'lon': -38.5267, 'name': 'Fortaleza'},
    'manaus': {'lat': -3.1190, 'lon': -60.0217, 'name': 'Manaus'},
    'recife': {'lat': -8.0476, 'lon': -34.8770, 'name': 'Recife'},
    'buenos_aires': {'lat': -34.6037, 'lon': -58.3816, 'name': 'Buenos Aires'},
    'cordoba_ar': {'lat': -31.4201, 'lon': -64.1888, 'name': 'Córdoba'},
    'rosario': {'lat': -32.9442, 'lon': -60.6505, 'name': 'Rosario'},
    'mendoza': {'lat': -32.8895, 'lon': -68.8458, 'name': 'Mendoza'},
    'ushuaia': {'lat': -54.8019, 'lon': -68.3030, 'name': 'Ushuaia'},
    'santiago': {'lat': -33.4489, 'lon': -70.6693, 'name': 'Santiago'},
    'valparaiso': {'lat': -33.0472, 'lon': -71.6127, 'name': 'Valparaíso'},
    'concepcion': {'lat': -36.8201, 'lon': -73.0444, 'name': 'Concepción'},
    'lima': {'lat': -12.0464, 'lon': -77.0428, 'name': 'Lima'},
    'cusco': {'lat': -13.5319, 'lon': -71.9675, 'name': 'Cusco'},
    'arequipa': {'lat': -16.4090, 'lon': -71.5375, 'name': 'Arequipa'},
    'bogota': {'lat': 4.7110, 'lon': -74.0721, 'name': 'Bogotá'},
    'medellin': {'lat': 6.2442, 'lon': -75.5812, 'name': 'Medellín'},
    'cali': {'lat': 3.4516, 'lon': -76.5320, 'name': 'Cali'},
    'quito': {'lat': -0.1807, 'lon': -78.4678, 'name': 'Quito'},
    'guayaquil': {'lat': -2.1709, 'lon': -79.9224, 'name': 'Guayaquil'},
    'la_paz': {'lat': -16.4897, 'lon': -68.1193, 'name': 'La Paz'},
    'santa_cruz_bo': {'lat': -17.7833, 'lon': -63.1821, 'name': 'Santa Cruz'},
    'asuncion': {'lat': -25.2637, 'lon': -57.5759, 'name': 'Asunción'},
    'montevideo': {'lat': -34.9011, 'lon': -56.1645, 'name': 'Montevideo'},
    'cape_town': {'lat': -33.9249, 'lon': 18.4241, 'name': 'Cape Town'},
    'johannesburg': {'lat': -26.2041, 'lon': 28.0473, 'name': 'Johannesburg'},
    'durban': {'lat': -29.8587, 'lon': 31.0218, 'name': 'Durban'},
    'pretoria': {'lat': -25.7479, 'lon': 28.2293, 'name': 'Pretoria'},
    'cairo': {'lat': 30.0444, 'lon': 31.2357, 'name': 'Cairo'},
    'alexandria': {'lat': 31.2001, 'lon': 29.9187, 'name': 'Alexandria'},
    'giza': {'lat': 30.0131, 'lon': 31.2089, 'name': 'Giza'},
    'casablanca': {'lat': 33.5731, 'lon': -7.5898, 'name': 'Casablanca'},
    'marrakesh': {'lat': 31.6295, 'lon': -7.9811, 'name': 'Marrakesh'},
    'lagos': {'lat': 6.5244, 'lon': 3.3792, 'name': 'Lagos'},
    'abuja': {'lat': 9.0765, 'lon': 7.3986, 'name': 'Abuja'},
    'nairobi': {'lat': -1.2921, 'lon': 36.8219, 'name': 'Nairobi'},
    'mombasa': {'lat': -4.0435, 'lon': 39.6682, 'name': 'Mombasa'},
    'addis_ababa': {'lat': 8.9806, 'lon': 38.7578, 'name': 'Addis Ababa'},
    'dar_es_salaam': {'lat': -6.7924, 'lon': 39.2083, 'name': 'Dar es Salaam'},
    'accra': {'lat': 5.6037, 'lon': -0.1870, 'name': 'Accra'},

    # 100 landscapes
    'himalayas': {'lat': 28.5983, 'lon': 83.9311, 'name': 'Himalayas'},
    'karakoram_range': {'lat': 35.8818, 'lon': 76.5133, 'name': 'Karakoram Range'},
    'hindu_kush': {'lat': 36.5000, 'lon': 71.5000, 'name': 'Hindu Kush'},
    'pamir_mountains': {'lat': 38.9000, 'lon': 73.5000, 'name': 'Pamir Mountains'},
    'tibetan_plateau': {'lat': 33.0000, 'lon': 88.0000, 'name': 'Tibetan Plateau'},
    'alps': {'lat': 46.8876, 'lon': 9.6570, 'name': 'Alps'},
    'pyrenees': {'lat': 42.6670, 'lon': 0.5000, 'name': 'Pyrenees'},
    'carpathian_mountains': {'lat': 48.7000, 'lon': 24.7000, 'name': 'Carpathian Mountains'},
    'andes': {'lat': -23.6500, 'lon': -67.7500, 'name': 'Andes'},
    'rocky_mountains': {'lat': 39.1130, 'lon': -106.4454, 'name': 'Rocky Mountains'},
    'appalachian_mountains': {'lat': 37.5000, 'lon': -81.0000, 'name': 'Appalachian Mountains'},
    'atlas_mountains': {'lat': 31.0000, 'lon': -7.0000, 'name': 'Atlas Mountains'},
    'great_dividing_range': {'lat': -36.5000, 'lon': 148.0000, 'name': 'Great Dividing Range'},
    'drakensberg': {'lat': -29.5000, 'lon': 29.2000, 'name': 'Drakensberg'},
    'ural_mountains': {'lat': 60.0000, 'lon': 60.0000, 'name': 'Ural Mountains'},
    'amazon_rainforest': {'lat': -3.4653, 'lon': -62.2159, 'name': 'Amazon Rainforest'},
    'congo_rainforest': {'lat': -1.5000, 'lon': 15.0000, 'name': 'Congo Rainforest'},
    'borneo_rainforest': {'lat': 0.9000, 'lon': 114.9000, 'name': 'Borneo Rainforest'},
    'sumatra_rainforest': {'lat': -0.5897, 'lon': 101.3431, 'name': 'Sumatra Rainforest'},
    'new_guinea_rainforest': {'lat': -5.0000, 'lon': 141.0000, 'name': 'New Guinea Rainforest'},
    'western_ghats': {'lat': 10.8505, 'lon': 76.2711, 'name': 'Western Ghats'},
    'eastern_himalayan_forest': {'lat': 27.8000, 'lon': 91.5000, 'name': 'Eastern Himalayan Forest'},
    'sundarbans': {'lat': 21.9497, 'lon': 89.1833, 'name': 'Sundarbans'},
    'taiga_siberia': {'lat': 61.0000, 'lon': 105.0000, 'name': 'Siberian Taiga'},
    'black_forest': {'lat': 48.0000, 'lon': 8.2000, 'name': 'Black Forest'},
    'yellowstone': {'lat': 44.4280, 'lon': -110.5885, 'name': 'Yellowstone National Park'},
    'yosemite': {'lat': 37.8651, 'lon': -119.5383, 'name': 'Yosemite National Park'},
    'banff': {'lat': 51.4968, 'lon': -115.9281, 'name': 'Banff National Park'},
    'kruger_national_park': {'lat': -23.9884, 'lon': 31.5547, 'name': 'Kruger National Park'},
    'serengeti': {'lat': -2.3333, 'lon': 34.8333, 'name': 'Serengeti National Park'},
    'sahara_desert': {'lat': 23.4162, 'lon': 25.6628, 'name': 'Sahara Desert'},
    'thar_desert': {'lat': 27.0000, 'lon': 71.0000, 'name': 'Thar Desert'},
    'gobi_desert': {'lat': 42.5903, 'lon': 103.4300, 'name': 'Gobi Desert'},
    'kalahari_desert': {'lat': -23.0000, 'lon': 22.0000, 'name': 'Kalahari Desert'},
    'simpson_desert': {'lat': -25.5000, 'lon': 137.5000, 'name': 'Simpson Desert'},
    'atacama_desert': {'lat': -24.5000, 'lon': -69.2500, 'name': 'Atacama Desert'},
    'mojave_desert': {'lat': 35.0110, 'lon': -115.4734, 'name': 'Mojave Desert'},
    'sonoran_desert': {'lat': 32.0000, 'lon': -112.0000, 'name': 'Sonoran Desert'},
    'great_victoria_desert': {'lat': -29.0000, 'lon': 129.0000, 'name': 'Great Victoria Desert'},
    'arabian_desert': {'lat': 23.0000, 'lon': 45.0000, 'name': 'Arabian Desert'},
    'death_valley': {'lat': 36.5323, 'lon': -116.9325, 'name': 'Death Valley'},
    'namib_desert': {'lat': -24.7500, 'lon': 15.3000, 'name': 'Namib Desert'},
    'salar_de_uyuni': {'lat': -20.1338, 'lon': -67.4891, 'name': 'Salar de Uyuni'},
    'vatnajokull': {'lat': 64.4167, 'lon': -16.6667, 'name': 'Vatnajökull Glacier'},
    'aletsch_glacier': {'lat': 46.4500, 'lon': 8.0500, 'name': 'Aletsch Glacier'},
    'great_barrier_reef': {'lat': -18.2871, 'lon': 147.6992, 'name': 'Great Barrier Reef'},
    'belize_barrier_reef': {'lat': 17.3150, 'lon': -87.5346, 'name': 'Belize Barrier Reef'},
    'red_sea_coral_reef': {'lat': 21.5000, 'lon': 39.0000, 'name': 'Red Sea Coral Reef'},
    'maldives_atolls': {'lat': 3.2028, 'lon': 73.2207, 'name': 'Maldives Atolls'},
    'galapagos_islands': {'lat': -0.9538, 'lon': -90.9656, 'name': 'Galápagos Islands'},
    'okinawa_reef': {'lat': 26.5000, 'lon': 127.9000, 'name': 'Okinawa Reef'},
    'komodo_national_park': {'lat': -8.5500, 'lon': 119.5000, 'name': 'Komodo National Park'},
    'fiordland': {'lat': -45.4170, 'lon': 167.7180, 'name': 'Fiordland National Park'},
    'everglades': {'lat': 25.2866, 'lon': -80.8987, 'name': 'Everglades'},
    'okavango_delta': {'lat': -19.3000, 'lon': 22.9000, 'name': 'Okavango Delta'},
    'pantanal': {'lat': -17.5000, 'lon': -57.0000, 'name': 'Pantanal'},
    'danube_delta': {'lat': 45.1667, 'lon': 29.3000, 'name': 'Danube Delta'},
    'mekong_delta': {'lat': 10.2000, 'lon': 105.9000, 'name': 'Mekong Delta'},
    'nile_delta': {'lat': 31.2500, 'lon': 31.0000, 'name': 'Nile Delta'},
    'orinoco_delta': {'lat': 8.8000, 'lon': -61.7000, 'name': 'Orinoco Delta'},
    'amazon_river': {'lat': -3.1190, 'lon': -60.0217, 'name': 'Amazon River'},
    'nile_river': {'lat': 15.6000, 'lon': 32.5000, 'name': 'Nile River'},
    'mississippi_river': {'lat': 35.1550, 'lon': -90.0659, 'name': 'Mississippi River'},
    'yangtze_river': {'lat': 30.5728, 'lon': 111.2908, 'name': 'Yangtze River'},
    'brahmaputra_river': {'lat': 26.2006, 'lon': 91.7362, 'name': 'Brahmaputra River'},
    'ganges_river': {'lat': 25.3176, 'lon': 83.0100, 'name': 'Ganges River'},
    'volga_river': {'lat': 48.7000, 'lon': 44.5000, 'name': 'Volga River'},
    'congo_river': {'lat': -4.3224, 'lon': 15.3070, 'name': 'Congo River'},
    'parana_river': {'lat': -27.5000, 'lon': -58.8000, 'name': 'Paraná River'},
    'murray_river': {'lat': -35.1167, 'lon': 139.2667, 'name': 'Murray River'},
    'lake_baikal': {'lat': 53.5587, 'lon': 108.1650, 'name': 'Lake Baikal'},
    'lake_victoria': {'lat': -1.0000, 'lon': 33.0000, 'name': 'Lake Victoria'},
    'lake_superior': {'lat': 47.7000, 'lon': -87.5000, 'name': 'Lake Superior'},
    'caspian_sea': {'lat': 41.7000, 'lon': 51.0000, 'name': 'Caspian Sea'},
    'dead_sea': {'lat': 31.5000, 'lon': 35.5000, 'name': 'Dead Sea'},
    'aral_sea': {'lat': 45.0000, 'lon': 60.0000, 'name': 'Aral Sea'},
    'great_salt_lake': {'lat': 41.1000, 'lon': -112.6000, 'name': 'Great Salt Lake'},
    'loch_ness': {'lat': 57.3229, 'lon': -4.4244, 'name': 'Loch Ness'},
    'crater_lake': {'lat': 42.9446, 'lon': -122.1090, 'name': 'Crater Lake'},
    'lake_titicaca': {'lat': -15.9254, 'lon': -69.3354, 'name': 'Lake Titicaca'},
    'iceland_volcanic_zone': {'lat': 64.0000, 'lon': -19.0000, 'name': 'Iceland Volcanic Zone'},
    'hawaii_volcanoes': {'lat': 19.4194, 'lon': -155.2885, 'name': 'Hawaii Volcanoes'},
    'mount_fuji': {'lat': 35.3606, 'lon': 138.7274, 'name': 'Mount Fuji'},
    'mount_kilimanjaro': {'lat': -3.0674, 'lon': 37.3556, 'name': 'Mount Kilimanjaro'},
    'mount_everest': {'lat': 27.9881, 'lon': 86.9250, 'name': 'Mount Everest'},
    'grand_canyon': {'lat': 36.1069, 'lon': -112.1129, 'name': 'Grand Canyon'},
    'bryce_canyon': {'lat': 37.5930, 'lon': -112.1871, 'name': 'Bryce Canyon'},
    'zion_canyon': {'lat': 37.2982, 'lon': -113.0263, 'name': 'Zion Canyon'},
    'uluru': {'lat': -25.3444, 'lon': 131.0369, 'name': 'Uluru'},
    'victoria_falls': {'lat': -17.9243, 'lon': 25.8572, 'name': 'Victoria Falls'},
    'iguazu_falls': {'lat': -25.6953, 'lon': -54.4367, 'name': 'Iguazú Falls'},
    'niagara_falls': {'lat': 43.0962, 'lon': -79.0377, 'name': 'Niagara Falls'},
    'cliffs_of_moher': {'lat': 52.9715, 'lon': -9.4309, 'name': 'Cliffs of Moher'},
    'white_cliffs_of_dover': {'lat': 51.1290, 'lon': 1.3210, 'name': 'White Cliffs of Dover'},
    'giants_causeway': {'lat': 55.2408, 'lon': -6.5116, 'name': "Giant's Causeway"},
    'patagonian_steppe': {'lat': -49.0000, 'lon': -70.0000, 'name': 'Patagonian Steppe'},
    'great_bear_lake': {'lat': 66.0000, 'lon': -121.0000, 'name': 'Great Bear Lake'},
    'okefenokee_swamp': {'lat': 30.7000, 'lon': -82.3000, 'name': 'Okefenokee Swamp'},
    'dolomites': {'lat': 46.4333, 'lon': 11.8500, 'name': 'Dolomites'},
    'bay_of_fundy': {'lat': 45.2500, 'lon': -64.5000, 'name': 'Bay of Fundy'},
}

TOTAL_REGIONS = len(REGIONS)

BANDS_RGB = ['SR_B4', 'SR_B3', 'SR_B2']
BANDS_TIR = ['ST_B10']

RADII = [5000, 8000, 12000, 20000, 30000, 40000]
SHIFTS = [(0, 0), (0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)]
DATE_RANGES = ["2024-01-01", "2023-01-01", "2022-01-01"]
CLOUD_COVERS = [10, 20, 35, 50, 70, 100]

PREPROCESS_SCALE = 3.0 / 20.0
SWATH_EDGE_THRESHOLD = 0.10

NETWORK_ERRORS = (
    ee.ee_exception.EEException,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
)


def setup_logger(log_file: Path) -> logging.Logger:
    """Configures the file logger without duplicating console logs."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('downloader')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(fh)
    return logger


def download_and_extract(url: str, dest_dir: Path, out_name: str) -> Path:
    """
    Downloads data from EE url and saves to dest_dir / out_name.
    Earth Engine may return a ZIP (PK\\x03\\x04) or a direct GeoTIFF.
    """
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    out_path = dest_dir / out_name

    # Check if the content is a ZIP file (magic number PK)
    if response.content[:4] == b'PK\x03\x04':
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            tif_files = [f for f in z.namelist() if f.endswith('.tif')]
            if not tif_files:
                raise ValueError("No .tif file found in downloaded zip.")

            # Extract the first tif file
            extracted_path = z.extract(tif_files[0], path=dest_dir)

            # Rename if necessary
            if Path(extracted_path).resolve() != out_path.resolve():
                Path(extracted_path).replace(out_path)
    else:
        # Save directly as tif
        with open(out_path, "wb") as f:
            f.write(response.content)

    return out_path


def verify_tiff(filepath: Path, expected_bands: int) -> None:
    """Verifies that the downloaded TIFF exists, is readable, and has expected bands."""
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} was not created.")

    try:
        with rasterio.open(filepath) as src:
            if src.count != expected_bands:
                raise ValueError(
                    f"Expected {expected_bands} bands, got {src.count}")
            if src.width == 0 or src.height == 0:
                raise ValueError("Image has zero width or height")
            if src.crs is None:
                raise ValueError("Image has no CRS defined")
    except rasterio.errors.RasterioIOError as e:
        raise ValueError(f"Corrupted GeoTIFF: {edef process_region(region_id: str, region_info: dict, output_dir: Path, overwrite: bool = False, verbose: bool = False) -> tuple[bool, str]:
    """Processes a single region with progressive fallback and adaptive radius expansion. Returns (success, message)."""
    dest_dir = output_dir / region_id
    rgb_path = dest_dir / "rgb.tif"
    tir_path = dest_dir / "tir.tif"

    if not overwrite and rgb_path.exists() and tir_path.exists():
        try:
            verify_tiff(rgb_path, 3)
            verify_tiff(tir_path, 1)
            return True, f"Skipping {region_info['name']} (already downloaded)"
        except Exception:
            pass  # Corrupted existing files, proceed to download and overwrite

    dest_dir.mkdir(parents=True, exist_ok=True)
    base_lat = region_info['lat']
    base_lon = region_info['lon']

    # Pre-flight radius filtering
    valid_radii = []
    for r in RADII:
        # 1 pixel = 30m, bbox width ≈ 2*r. Scale to preprocessing grid.
        est_grid = int((2 * r / 30.0) * PREPROCESS_SCALE)
        if est_grid >= 80:
            valid_radii.append(r)
            
    # Ensure minimum radius is mathematically valid if none exist
    if not valid_radii or min(valid_radii) > 10000:
        valid_radii = sorted(list(set([8000] + valid_radii)))

    attempt = 0
    total_attempts = len(valid_radii) * len(SHIFTS) * \
        len(DATE_RANGES) * len(CLOUD_COVERS)

    for base_radius in valid_radii:
        for shift_lat, shift_lon in SHIFTS:
            for date_start in DATE_RANGES:
                for cloud in CLOUD_COVERS:
                    attempt += 1
                    lat = base_lat + shift_lat
                    lon = base_lon + shift_lon
                    
                    current_radius = base_radius

                    # Adaptive download loop
                    while current_radius <= 40000:
                        if verbose:
                            print(f"\nAttempt {attempt}/{total_attempts} (Radius: {current_radius//1000} km)")
                            print(f"Region : {region_info['name']}")
                            print(f"Date   : {date_start} → Present")
                            print(f"Cloud  : ≤{cloud}%")
                            print(f"Shift  : ({shift_lat}, {shift_lon})")

                        point = ee.Geometry.Point([lon, lat])
                        region = point.buffer(current_radius).bounds()

                        collection = (
                            ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                            .filterBounds(point)
                            .filterDate(date_start, "2024-06-30")
                        )

                        if cloud < 100:
                            collection = collection.filter(
                                ee.Filter.lt('CLOUD_COVER', cloud))

                        collection = (
                            collection
                            .sort('CLOUD_COVER')
                            .sort('system:time_start', False)
                        )

                        try:
                            count = collection.size().getInfo()
                        except NETWORK_ERRORS as e:
                            if verbose:
                                print(f"Result : Network Error ({type(e).__name__})")
                            raise  # Re-raise to trigger exponential backoff in worker
                        except Exception as e:
                            if verbose:
                                print(f"Result : EE Error ({e})")
                            break # Break inner radius loop, continue outer loops

                        if count == 0:
                            if verbose:
                                print("Result : No imagery")
                            break # Break inner radius loop, continue outer loops

                        if verbose:
                            print("Result : Imagery found! Downloading...")

                        try:
                            image = collection.first().clip(region)
                            rgb_url = image.select(BANDS_RGB).getDownloadURL(
                                {'scale': 30, 'region': region, 'format': 'GEO_TIFF'})
                            download_and_extract(rgb_url, dest_dir, "rgb.tif")
                            verify_tiff(rgb_path, 3)

                            tir_url = image.select(BANDS_TIR).getDownloadURL(
                                {'scale': 30, 'region': region, 'format': 'GEO_TIFF'})
                            download_and_extract(tir_url, dest_dir, "tir.tif")
                            verify_tiff(tir_path, 1)
                            
                            # Post-download verification of actual image size and valid data
                            with rasterio.open(rgb_path) as src:
                                w, h = src.width, src.height
                                # Use GDAL dataset mask for robust NoData detection across all bands
                                mask = src.dataset_mask()
                                valid_ratio = np.count_nonzero(mask) / mask.size
                                
                            grid_w = round(w * PREPROCESS_SCALE)
                            grid_h = round(h * PREPROCESS_SCALE)
                            
                            if grid_w < 64 or grid_h < 64:
                                if verbose:
                                    print(f"\nRegion: {region_info['name']}")
                                    print(f"Grid: {grid_w}x{grid_h}")
                                    print(f"Valid pixels: {valid_ratio*100:.0f}%")
                                    print("Decision:")
                                
                                # Clean up files so we can retry
                                for f in ["rgb.tif", "tir.tif"]:
                                    fpath = dest_dir / f
                                    if fpath.exists():
                                        with suppress(Exception):
                                            fpath.unlink()
                                            
                                # Decision tree for recovery
                                if (1.0 - valid_ratio) > SWATH_EDGE_THRESHOLD:
                                    # Swath-edge issue: lots of NoData. A larger radius won't help the same scene.
                                    if verbose:
                                        print("Likely swath edge. Trying next acquisition.")
                                    break # Give up on this scene, let outer loops try other shifts/dates
                                else:
                                    # Geographic extent issue: image is small but mostly valid. Increase radius.
                                    if current_radius < 40000:
                                        if verbose:
                                            print(f"Increasing radius to {(current_radius + 5000)//1000} km.")
                                        current_radius += 5000
                                        continue
                                    else:
                                        if verbose:
                                            print("Maximum radius reached, scene still too small.")
                                        break

                            if verbose:
                                print("Result : ✅ Success")
                            return True, f"Successfully downloaded {region_info['name']} on attempt {attempt}"

                        except NETWORK_ERRORS as e:
                            raise  # Trigger network backoff
                        except Exception as e:
                            if verbose:
                                print(f"Result : Download/Verify Failed ({e})")
                            # Cleanup broken files for this attempt
                            for f in ["rgb.tif", "tir.tif"]:
                                fpath = dest_dir / f
                                if fpath.exists():
                                    with suppress(Exception):
                                        fpath.unlink()
                            break # Break inner radius loop, continue outer loops

    return False, f"No imagery found for {region_info['name']} after {total_attempts} attempts"al_attempts} attempts"


def worker(task: tuple) -> tuple:
    """Worker function for threading. Handles network retries."""
    region_id, region_info, output_dir, overwrite, verbose = task

    for attempt in range(1, 6):  # Max 5 retries for network errors
        try:
            success, msg = process_region(
                region_id, region_info, output_dir, overwrite, verbose)
            return region_id, success, msg, attempt, False
        except NETWORK_ERRORS as e:
            if attempt < 5:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s, 8s
            else:
                return region_id, False, f"Network error after 5 retries: {e}", attempt, True
        except Exception as e:
            return region_id, False, f"Unexpected error: {e}", attempt, False

    return region_id, False, "Unknown error", 5, False


def save_progress(state: dict, filepath: Path):
    import json
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)


def main():
    import json
    parser = argparse.ArgumentParser(
        description="Direct Landsat 9 Downloader (Local)")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT /
                        "data/landsat9/raw"), help="Output directory")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel downloads")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every parameter combination attempted")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(Path("logs/download.log"))
    logger.info("Starting Landsat 9 direct local downloader")

    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception as exc:
        print(f"Failed to initialize Earth Engine: {exc}")
        logger.error(f"Earth Engine init failed: {exc}")
        return

    # Initialize State
    state_file = Path("progress.json")
    state = {
        "completed": 0,
        "successful_regions": [],
        "failed_regions": [],
        "network_errors": []
    }

    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                state.update(saved)
                print(
                    f"Resumed from checkpoint: {state['completed']} completed, {len(state['successful_regions'])} successful")
        except Exception:
            print("Failed to load progress.json, starting fresh.")

    def run_pass(regions_to_run, pass_name):
        print(f"\\n--- {pass_name} ---")
        if not regions_to_run:
            print("No regions to process in this pass.")
            return

        tasks = [(rid, REGIONS[rid], output_dir, args.overwrite, args.verbose)
                 for rid in regions_to_run]
        total = len(tasks)
        success_in_pass = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(worker, t): t for t in tasks}
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    region_id, success, msg, attempts, is_network = future.result()

                    state["completed"] += 1

                    if success:
                        success_in_pass += 1
                        if region_id not in state["successful_regions"]:
                            state["successful_regions"].append(region_id)
                        if region_id in state["failed_regions"]:
                            state["failed_regions"].remove(region_id)
                        if region_id in state["network_errors"]:
                            state["network_errors"].remove(region_id)

                        if not args.verbose:
                            print(
                                f"[{i}/{total}] ✅ {REGIONS[region_id]['name']}")
                    else:
                        if is_network:
                            if region_id not in state["network_errors"]:
                                state["network_errors"].append(region_id)
                            if not args.verbose:
                                print(
                                    f"[{i}/{total}] 🌐 Network Error: {REGIONS[region_id]['name']}")
                        else:
                            if region_id not in state["failed_regions"]:
                                state["failed_regions"].append(region_id)
                            if not args.verbose:
                                print(
                                    f"[{i}/{total}] ❌ Failed: {REGIONS[region_id]['name']} - {msg}")

                    if success_in_pass > 0 and success_in_pass % 10 == 0:
                        save_progress(state, state_file)

        except KeyboardInterrupt:
            print("\\nInterrupted by user! Saving progress...")
            save_progress(state, state_file)
            sys.exit(0)

        save_progress(state, state_file)

    # Determine regions to run
    all_regions = list(REGIONS.keys())
    pass1_regions = [
        r for r in all_regions if r not in state["successful_regions"]]

    start_time = time.time()

    # Pass 1
    if pass1_regions:
        run_pass(pass1_regions, "PASS 1")

    # Pass 2
    pass2_regions = state["failed_regions"].copy() + \
        state["network_errors"].copy()
    if pass2_regions:
        run_pass(pass2_regions, "PASS 2")

    # Pass 3
    pass3_regions = state["failed_regions"].copy() + \
        state["network_errors"].copy()
    if pass3_regions:
        run_pass(pass3_regions, "PASS 3")

    # Final Summary
    print("\\nFINAL")
    print("--------")
    print(f"Successful : {len(state['successful_regions'])}")
    print(f"Failed     : {len(state['failed_regions'])}")
    print(f"Network    : {len(state['network_errors'])}")

    elapsed = time.time() - start_time
    print(f"Time        : {datetime.timedelta(seconds=int(elapsed))}")

    if state["failed_regions"] or state["network_errors"]:
        print("\\nRemaining unavailable regions:")
        for r in state["failed_regions"] + state["network_errors"]:
            print(f"- {REGIONS[r]['name']}")

    # Write individual state lists
    with open("successful_regions.json", "w") as f:
        json.dump(state["successful_regions"], f)
    with open("failed_regions.json", "w") as f:
        json.dump(state["failed_regions"], f)
    with open("network_errors.json", "w") as f:
        json.dump(state["network_errors"], f)


if __name__ == '__main__':
    main()
