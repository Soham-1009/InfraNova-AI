"""
Download Landsat 9 TIR + RGB data via Google Drive export.
"""

import ee
import time
from pathlib import Path

EE_PROJECT_ID = "infranova-ai"  # Update if your project ID differs

REGIONS = {
    # 100 Indian cities
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
    'amritsar': {'lat': 31.6340, 'lon': 74.8723, 'name': 'Amritsar'},
    'varanasi': {'lat': 25.3176, 'lon': 82.9739, 'name': 'Varanasi'},
    'agra': {'lat': 27.1767, 'lon': 78.0081, 'name': 'Agra'},
    'thiruvananthapuram': {'lat': 8.5241, 'lon': 76.9366, 'name': 'Thiruvananthapuram'},
    'goa_panaji': {'lat': 15.4909, 'lon': 73.8278, 'name': 'Panaji Goa'},
    'mangalore': {'lat': 12.9141, 'lon': 74.8560, 'name': 'Mangalore'},
    'mysore': {'lat': 12.2958, 'lon': 76.6394, 'name': 'Mysore'},
    'coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'name': 'Coimbatore'},
    'madurai': {'lat': 9.9252, 'lon': 78.1198, 'name': 'Madurai'},
    'vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'name': 'Vijayawada'},
    'dehradun': {'lat': 30.3165, 'lon': 78.0322, 'name': 'Dehradun'},
    'shimla': {'lat': 31.1048, 'lon': 77.1734, 'name': 'Shimla'},
    'srinagar': {'lat': 34.0837, 'lon': 74.7973, 'name': 'Srinagar'},
    'gangtok': {'lat': 27.3389, 'lon': 88.6065, 'name': 'Gangtok'},
    'jodhpur': {'lat': 26.2389, 'lon': 73.0243, 'name': 'Jodhpur'},
    'udaipur': {'lat': 24.5854, 'lon': 73.7125, 'name': 'Udaipur'},
    'rajkot': {'lat': 22.3039, 'lon': 70.8022, 'name': 'Rajkot'},
    'jamshedpur': {'lat': 22.8046, 'lon': 86.2029, 'name': 'Jamshedpur'},
    'ranchi': {'lat': 23.3441, 'lon': 85.3096, 'name': 'Ranchi'},
    'raipur': {'lat': 21.2514, 'lon': 81.6296, 'name': 'Raipur'},
    'meerut': {'lat': 28.9845, 'lon': 77.7064, 'name': 'Meerut'},
    'aligarh': {'lat': 27.8974, 'lon': 78.0880, 'name': 'Aligarh'},
    'allahabad': {'lat': 25.4358, 'lon': 81.8463, 'name': 'Prayagraj'},
    'gorakhpur': {'lat': 26.7606, 'lon': 83.3732, 'name': 'Gorakhpur'},
    'noida': {'lat': 28.5355, 'lon': 77.3910, 'name': 'Noida'},
    'gurgaon': {'lat': 28.4595, 'lon': 77.0266, 'name': 'Gurgaon'},
    'faridabad': {'lat': 28.4089, 'lon': 77.3178, 'name': 'Faridabad'},
    'ghaziabad': {'lat': 28.6692, 'lon': 77.4538, 'name': 'Ghaziabad'},
    'howrah': {'lat': 22.5958, 'lon': 88.2636, 'name': 'Howrah'},
    'durgapur': {'lat': 23.5204, 'lon': 87.3119, 'name': 'Durgapur'},
    'asansol': {'lat': 23.6849, 'lon': 86.9661, 'name': 'Asansol'},
    'siliguri': {'lat': 26.7271, 'lon': 88.3953, 'name': 'Siliguri'},
    'bhubaneswar': {'lat': 20.2961, 'lon': 85.8245, 'name': 'Bhubaneswar'},
    'cuttack': {'lat': 20.4625, 'lon': 85.8830, 'name': 'Cuttack'},
    'puri': {'lat': 19.8135, 'lon': 85.8312, 'name': 'Puri'},
    'rourkela': {'lat': 22.2604, 'lon': 84.8536, 'name': 'Rourkela'},
    'aurangabad': {'lat': 19.8762, 'lon': 75.3433, 'name': 'Aurangabad'},
    'nashik': {'lat': 19.9975, 'lon': 73.7898, 'name': 'Nashik'},
    'solapur': {'lat': 17.6599, 'lon': 75.9064, 'name': 'Solapur'},
    'kolhapur': {'lat': 16.7050, 'lon': 74.2433, 'name': 'Kolhapur'},
    'sangli': {'lat': 16.8524, 'lon': 74.5815, 'name': 'Sangli'},
    'thane': {'lat': 19.2183, 'lon': 72.9781, 'name': 'Thane'},
    'navi_mumbai': {'lat': 19.0330, 'lon': 73.0297, 'name': 'Navi Mumbai'},
    'vadodara': {'lat': 22.3072, 'lon': 73.1812, 'name': 'Vadodara'},
    'bhavnagar': {'lat': 21.7645, 'lon': 72.1519, 'name': 'Bhavnagar'},
    'jamnagar': {'lat': 22.4707, 'lon': 70.0577, 'name': 'Jamnagar'},
    'gandhinagar': {'lat': 23.2156, 'lon': 72.6369, 'name': 'Gandhinagar'},
    'kota': {'lat': 25.2138, 'lon': 75.8648, 'name': 'Kota'},
    'ajmer': {'lat': 26.4499, 'lon': 74.6399, 'name': 'Ajmer'},
    'bikaner': {'lat': 28.0229, 'lon': 73.3119, 'name': 'Bikaner'},
    'jaisalmer': {'lat': 26.9157, 'lon': 70.9083, 'name': 'Jaisalmer'},
    'mount_abu': {'lat': 24.5926, 'lon': 72.7156, 'name': 'Mount Abu'},
    'pushkar': {'lat': 26.4900, 'lon': 74.5500, 'name': 'Pushkar'},
    'bhilai': {'lat': 21.2092, 'lon': 81.4285, 'name': 'Bhilai'},
    'bilaspur': {'lat': 22.0797, 'lon': 82.1409, 'name': 'Bilaspur'},
    'jabalpur': {'lat': 23.1815, 'lon': 79.9864, 'name': 'Jabalpur'},
    'gwalior': {'lat': 26.2183, 'lon': 78.1828, 'name': 'Gwalior'},
    'ujjain': {'lat': 23.1765, 'lon': 75.7885, 'name': 'Ujjain'},
    'sagar': {'lat': 23.8388, 'lon': 78.7378, 'name': 'Sagar'},
    'satna': {'lat': 24.5800, 'lon': 80.8300, 'name': 'Satna'},
    'rewa': {'lat': 24.5350, 'lon': 81.3050, 'name': 'Rewa'},
    'tiruchirappalli': {'lat': 10.7905, 'lon': 78.7047, 'name': 'Tiruchirappalli'},
    'salem': {'lat': 11.6643, 'lon': 78.1460, 'name': 'Salem'},
    'tirupati': {'lat': 13.6288, 'lon': 79.4192, 'name': 'Tirupati'},
    'tirupur': {'lat': 11.1085, 'lon': 77.3411, 'name': 'Tirupur'},
    'vellore': {'lat': 12.9165, 'lon': 79.1325, 'name': 'Vellore'},
    'pondicherry': {'lat': 11.9416, 'lon': 79.8083, 'name': 'Pondicherry'},
    'rameshwaram': {'lat': 9.2876, 'lon': 79.3129, 'name': 'Rameshwaram'},
    'kanyakumari': {'lat': 8.0883, 'lon': 77.5385, 'name': 'Kanyakumari'},
    'guntur': {'lat': 16.3067, 'lon': 80.4365, 'name': 'Guntur'},
    'nellore': {'lat': 14.4426, 'lon': 79.9865, 'name': 'Nellore'},
    'kurnool': {'lat': 15.8281, 'lon': 78.0373, 'name': 'Kurnool'},
    'warangal': {'lat': 17.9689, 'lon': 79.5941, 'name': 'Warangal'},
    'belgaum': {'lat': 15.8497, 'lon': 74.4977, 'name': 'Belgaum'},
    'hubli': {'lat': 15.3647, 'lon': 75.1240, 'name': 'Hubli'},
    'davangere': {'lat': 14.4644, 'lon': 75.9218, 'name': 'Davangere'},
    'gulbarga': {'lat': 17.3297, 'lon': 76.8343, 'name': 'Gulbarga'},
    'shimoga': {'lat': 13.9299, 'lon': 75.5681, 'name': 'Shimoga'},
    'hampi': {'lat': 15.3350, 'lon': 76.4600, 'name': 'Hampi'},
    'kozhikode': {'lat': 11.2588, 'lon': 75.7804, 'name': 'Kozhikode'},
    'thrissur': {'lat': 10.5276, 'lon': 76.2144, 'name': 'Thrissur'},
    'kollam': {'lat': 8.8932, 'lon': 76.6141, 'name': 'Kollam'},
    'alappuzha': {'lat': 9.4981, 'lon': 76.3388, 'name': 'Alappuzha'},
    'kannur': {'lat': 11.8745, 'lon': 75.3704, 'name': 'Kannur'},
    'jalandhar': {'lat': 31.3260, 'lon': 75.5762, 'name': 'Jalandhar'},
    'ludhiana': {'lat': 30.9010, 'lon': 75.8573, 'name': 'Ludhiana'},
    'patiala': {'lat': 30.3398, 'lon': 76.3869, 'name': 'Patiala'},
    'pathankot': {'lat': 32.2643, 'lon': 75.6421, 'name': 'Pathankot'},
    'panipat': {'lat': 29.3909, 'lon': 76.9635, 'name': 'Panipat'},
    'rohtak': {'lat': 28.8955, 'lon': 76.6066, 'name': 'Rohtak'},
    'hisar': {'lat': 29.1492, 'lon': 75.7217, 'name': 'Hisar'},
    'karnal': {'lat': 29.6857, 'lon': 76.9905, 'name': 'Karnal'},
    'manali': {'lat': 32.2396, 'lon': 77.1887, 'name': 'Manali'},
    'leh': {'lat': 34.1526, 'lon': 77.5770, 'name': 'Leh'},
    'darjeeling': {'lat': 27.0410, 'lon': 88.2663, 'name': 'Darjeeling'},
    'shillong': {'lat': 25.5788, 'lon': 91.8933, 'name': 'Shillong'},
    'imphal': {'lat': 24.8170, 'lon': 93.9368, 'name': 'Imphal'},
    'aizawl': {'lat': 23.7271, 'lon': 92.7176, 'name': 'Aizawl'},
    'agartala': {'lat': 23.8315, 'lon': 91.2868, 'name': 'Agartala'},
    'kohima': {'lat': 25.6747, 'lon': 94.1086, 'name': 'Kohima'},
    'itanagar': {'lat': 27.0844, 'lon': 93.6053, 'name': 'Itanagar'},
    'dispur': {'lat': 26.1433, 'lon': 91.7898, 'name': 'Dispur'},
    'port_blair': {'lat': 11.6234, 'lon': 92.7265, 'name': 'Port Blair'},
    'silvassa': {'lat': 20.2738, 'lon': 73.0140, 'name': 'Silvassa'},
    'daman': {'lat': 20.3974, 'lon': 72.8328, 'name': 'Daman'},
    
    # 90 international cities
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
    'sao_paulo': {'lat': -23.5505, 'lon': -46.6333, 'name': 'Sao Paulo'},
    'lima': {'lat': -12.0464, 'lon': -77.0428, 'name': 'Lima'},
    'bogota': {'lat': 4.7110, 'lon': -74.0721, 'name': 'Bogota'},
    'santiago': {'lat': -33.4489, 'lon': -70.6693, 'name': 'Santiago'},
    'caracas': {'lat': 10.4806, 'lon': -66.9036, 'name': 'Caracas'},
    'havana': {'lat': 23.1136, 'lon': -82.3666, 'name': 'Havana'},
    'panama_city': {'lat': 8.9824, 'lon': -79.5199, 'name': 'Panama City'},
    'chicago': {'lat': 41.8781, 'lon': -87.6298, 'name': 'Chicago'},
    'houston': {'lat': 29.7604, 'lon': -95.3698, 'name': 'Houston'},
    'phoenix': {'lat': 33.4484, 'lon': -112.0740, 'name': 'Phoenix'},
    'philadelphia': {'lat': 39.9526, 'lon': -75.1652, 'name': 'Philadelphia'},
    'san_diego': {'lat': 32.7157, 'lon': -117.1611, 'name': 'San Diego'},
    'dallas': {'lat': 32.7767, 'lon': -96.7970, 'name': 'Dallas'},
    'austin': {'lat': 30.2672, 'lon': -97.7431, 'name': 'Austin'},
    'seattle': {'lat': 47.6062, 'lon': -122.3321, 'name': 'Seattle'},
    'boston': {'lat': 42.3601, 'lon': -71.0589, 'name': 'Boston'},
    'miami': {'lat': 25.7617, 'lon': -80.1918, 'name': 'Miami'},
    'denver': {'lat': 39.7392, 'lon': -104.9903, 'name': 'Denver'},
    'atlanta': {'lat': 33.7490, 'lon': -84.3880, 'name': 'Atlanta'},
    'washington_dc': {'lat': 38.9072, 'lon': -77.0369, 'name': 'Washington DC'},
    'montreal': {'lat': 45.5017, 'lon': -73.5673, 'name': 'Montreal'},
    'vancouver': {'lat': 49.2827, 'lon': -123.1207, 'name': 'Vancouver'},
    'calgary': {'lat': 51.0447, 'lon': -114.0719, 'name': 'Calgary'},
    'ottawa': {'lat': 45.4215, 'lon': -75.6972, 'name': 'Ottawa'},
    'madrid': {'lat': 40.4168, 'lon': -3.7038, 'name': 'Madrid'},
    'barcelona': {'lat': 41.3851, 'lon': 2.1734, 'name': 'Barcelona'},
    'rome': {'lat': 41.9028, 'lon': 12.4964, 'name': 'Rome'},
    'milan': {'lat': 45.4642, 'lon': 9.1900, 'name': 'Milan'},
    'berlin': {'lat': 52.5200, 'lon': 13.4050, 'name': 'Berlin'},
    'munich': {'lat': 48.1351, 'lon': 11.5820, 'name': 'Munich'},
    'amsterdam': {'lat': 52.3676, 'lon': 4.9041, 'name': 'Amsterdam'},
    'brussels': {'lat': 50.8503, 'lon': 4.3517, 'name': 'Brussels'},
    'vienna': {'lat': 48.2082, 'lon': 16.3738, 'name': 'Vienna'},
    'zurich': {'lat': 47.3769, 'lon': 8.5417, 'name': 'Zurich'},
    'stockholm': {'lat': 59.3293, 'lon': 18.0686, 'name': 'Stockholm'},
    'oslo': {'lat': 59.9139, 'lon': 10.7522, 'name': 'Oslo'},
    'copenhagen': {'lat': 55.6761, 'lon': 12.5683, 'name': 'Copenhagen'},
    'helsinki': {'lat': 60.1699, 'lon': 24.9384, 'name': 'Helsinki'},
    'warsaw': {'lat': 52.2297, 'lon': 21.0122, 'name': 'Warsaw'},
    'prague': {'lat': 50.0755, 'lon': 14.4378, 'name': 'Prague'},
    'budapest': {'lat': 47.4979, 'lon': 19.0402, 'name': 'Budapest'},
    'athens': {'lat': 37.9838, 'lon': 23.7275, 'name': 'Athens'},
    'lisbon': {'lat': 38.7223, 'lon': -9.1393, 'name': 'Lisbon'},
    'dublin': {'lat': 53.3498, 'lon': -6.2603, 'name': 'Dublin'},
    'edinburgh': {'lat': 55.9533, 'lon': -3.1883, 'name': 'Edinburgh'},
    'manchester': {'lat': 53.4808, 'lon': -2.2426, 'name': 'Manchester'},
    'reykjavik': {'lat': 64.1466, 'lon': -21.9426, 'name': 'Reykjavik'},
    'tehran': {'lat': 35.6892, 'lon': 51.3890, 'name': 'Tehran'},
    'baghdad': {'lat': 33.3152, 'lon': 44.3661, 'name': 'Baghdad'},
    'riyadh': {'lat': 24.7136, 'lon': 46.6753, 'name': 'Riyadh'},
    'doha': {'lat': 25.2854, 'lon': 51.5310, 'name': 'Doha'},
    'kuwait_city': {'lat': 29.3759, 'lon': 47.9774, 'name': 'Kuwait City'},
    'amman': {'lat': 31.9454, 'lon': 35.9284, 'name': 'Amman'},
    'beirut': {'lat': 33.8938, 'lon': 35.5018, 'name': 'Beirut'},
    'jerusalem': {'lat': 31.7683, 'lon': 35.2137, 'name': 'Jerusalem'},
    'karachi': {'lat': 24.8607, 'lon': 67.0011, 'name': 'Karachi'},
    'lahore': {'lat': 31.5497, 'lon': 74.3436, 'name': 'Lahore'},
    'islamabad': {'lat': 33.6844, 'lon': 73.0479, 'name': 'Islamabad'},
    'dhaka': {'lat': 23.8103, 'lon': 90.4125, 'name': 'Dhaka'},
    'kathmandu': {'lat': 27.7172, 'lon': 85.3240, 'name': 'Kathmandu'},
    'colombo': {'lat': 6.9271, 'lon': 79.8612, 'name': 'Colombo'},
    'yangon': {'lat': 16.8409, 'lon': 96.1735, 'name': 'Yangon'},
    'hanoi': {'lat': 21.0285, 'lon': 105.8542, 'name': 'Hanoi'},
    'ho_chi_minh': {'lat': 10.8231, 'lon': 106.6297, 'name': 'Ho Chi Minh'},
    'manila': {'lat': 14.5995, 'lon': 120.9842, 'name': 'Manila'},
    'jakarta': {'lat': -6.2088, 'lon': 106.8456, 'name': 'Jakarta'},
    'kuala_lumpur': {'lat': 3.1390, 'lon': 101.6869, 'name': 'Kuala Lumpur'},
    'shanghai': {'lat': 31.2304, 'lon': 121.4737, 'name': 'Shanghai'},
    'hong_kong': {'lat': 22.3193, 'lon': 114.1694, 'name': 'Hong Kong'},
    'osaka': {'lat': 34.6937, 'lon': 135.5023, 'name': 'Osaka'},
    'nairobi': {'lat': -1.2921, 'lon': 36.8219, 'name': 'Nairobi'},
    'addis_ababa': {'lat': 9.0320, 'lon': 38.7469, 'name': 'Addis Ababa'},
    'casablanca': {'lat': 33.5731, 'lon': -7.5898, 'name': 'Casablanca'},
    'melbourne': {'lat': -37.8136, 'lon': 144.9631, 'name': 'Melbourne'},
    'perth': {'lat': -31.9505, 'lon': 115.8605, 'name': 'Perth'},
    'auckland': {'lat': -36.8485, 'lon': 174.7633, 'name': 'Auckland'},
    
    # 60 landscapes
    'amazon': {'lat': -3.4653, 'lon': -62.2159, 'name': 'Amazon'},
    'sahara': {'lat': 23.4162, 'lon': 25.6628, 'name': 'Sahara'},
    'himalayas': {'lat': 27.9881, 'lon': 86.9250, 'name': 'Himalayas'},
    'ganges': {'lat': 25.3176, 'lon': 82.9739, 'name': 'Ganges'},
    'sundarbans': {'lat': 21.9497, 'lon': 89.1833, 'name': 'Sundarbans'},
    'andes': {'lat': -13.5320, 'lon': -71.9675, 'name': 'Andes'},
    'alps': {'lat': 46.5197, 'lon': 8.5500, 'name': 'Alps'},
    'congo': {'lat': -2.0, 'lon': 23.0, 'name': 'Congo'},
    'siberia': {'lat': 60.0, 'lon': 100.0, 'name': 'Siberia'},
    'gobi': {'lat': 42.0, 'lon': 105.0, 'name': 'Gobi'},
    'western_ghats': {'lat': 13.0, 'lon': 75.0, 'name': 'Western Ghats'},
    'eastern_ghats': {'lat': 17.0, 'lon': 82.0, 'name': 'Eastern Ghats'},
    'aravalli': {'lat': 25.0, 'lon': 73.0, 'name': 'Aravalli Range'},
    'deccan_plateau': {'lat': 17.0, 'lon': 77.0, 'name': 'Deccan Plateau'},
    'thar_desert': {'lat': 27.0, 'lon': 71.0, 'name': 'Thar Desert'},
    'rann_kutch': {'lat': 24.0, 'lon': 70.5, 'name': 'Rann of Kutch'},
    'godavari_delta': {'lat': 16.5, 'lon': 81.8, 'name': 'Godavari Delta'},
    'indo_gangetic': {'lat': 27.0, 'lon': 80.0, 'name': 'Indo-Gangetic Plain'},
    'nilgiri_hills': {'lat': 11.4, 'lon': 76.7, 'name': 'Nilgiri Hills'},
    'satpura': {'lat': 22.0, 'lon': 78.5, 'name': 'Satpura Range'},
    'vindhya': {'lat': 24.0, 'lon': 80.0, 'name': 'Vindhya Range'},
    'kaziranga': {'lat': 26.5, 'lon': 93.4, 'name': 'Kaziranga'},
    'bandipur': {'lat': 11.7, 'lon': 76.6, 'name': 'Bandipur Forest'},
    'periyar': {'lat': 9.5, 'lon': 77.2, 'name': 'Periyar'},
    'corbett': {'lat': 29.5, 'lon': 78.9, 'name': 'Jim Corbett'},
    'ranthambore': {'lat': 26.0, 'lon': 76.4, 'name': 'Ranthambore'},
    'rockies': {'lat': 39.0, 'lon': -106.0, 'name': 'Rocky Mountains'},
    'pyrenees': {'lat': 42.6, 'lon': 1.0, 'name': 'Pyrenees'},
    'urals': {'lat': 60.0, 'lon': 60.0, 'name': 'Ural Mountains'},
    'caucasus': {'lat': 43.0, 'lon': 43.0, 'name': 'Caucasus'},
    'atlas_mountains': {'lat': 31.0, 'lon': -7.0, 'name': 'Atlas Mountains'},
    'tibetan_plateau': {'lat': 32.0, 'lon': 88.0, 'name': 'Tibetan Plateau'},
    'great_lakes': {'lat': 45.0, 'lon': -85.0, 'name': 'Great Lakes'},
    'lake_baikal': {'lat': 53.0, 'lon': 108.0, 'name': 'Lake Baikal'},
    'caspian_sea': {'lat': 41.0, 'lon': 51.0, 'name': 'Caspian Sea'},
    'dead_sea': {'lat': 31.5, 'lon': 35.5, 'name': 'Dead Sea'},
    'nile_delta': {'lat': 31.0, 'lon': 31.0, 'name': 'Nile Delta'},
    'mekong_delta': {'lat': 10.0, 'lon': 105.5, 'name': 'Mekong Delta'},
    'mississippi_delta': {'lat': 29.5, 'lon': -89.5, 'name': 'Mississippi Delta'},
    'kalahari': {'lat': -24.0, 'lon': 21.5, 'name': 'Kalahari Desert'},
    'patagonia': {'lat': -45.0, 'lon': -70.0, 'name': 'Patagonia'},
    'atacama': {'lat': -24.0, 'lon': -69.0, 'name': 'Atacama Desert'},
    'mojave': {'lat': 35.0, 'lon': -116.0, 'name': 'Mojave Desert'},
    'namib': {'lat': -23.0, 'lon': 15.0, 'name': 'Namib Desert'},
    'borneo_forest': {'lat': 1.0, 'lon': 114.0, 'name': 'Borneo Rainforest'},
    'serengeti': {'lat': -2.0, 'lon': 35.0, 'name': 'Serengeti'},
    'yellowstone': {'lat': 44.5, 'lon': -110.5, 'name': 'Yellowstone'},
    'grand_canyon': {'lat': 36.0, 'lon': -112.0, 'name': 'Grand Canyon'},
    'iceland_glaciers': {'lat': 64.0, 'lon': -19.0, 'name': 'Iceland Glaciers'},
    'mount_fuji': {'lat': 35.4, 'lon': 138.7, 'name': 'Mount Fuji'},
    'kilimanjaro': {'lat': -3.1, 'lon': 37.4, 'name': 'Kilimanjaro'},
    'great_barrier': {'lat': -18.0, 'lon': 147.0, 'name': 'Great Barrier Reef'},
    'maldives_atolls': {'lat': 3.0, 'lon': 73.0, 'name': 'Maldives'},
    'galapagos': {'lat': -1.0, 'lon': -90.0, 'name': 'Galapagos'},
    'hawaii': {'lat': 20.0, 'lon': -156.0, 'name': 'Hawaii'},
    'norwegian_fjords': {'lat': 60.5, 'lon': 7.0, 'name': 'Norwegian Fjords'},
    'taiga': {'lat': 60.0, 'lon': 80.0, 'name': 'Siberian Taiga'},
    'pampas': {'lat': -36.0, 'lon': -62.0, 'name': 'Pampas'},
    'greenland_ice': {'lat': 72.0, 'lon': -42.0, 'name': 'Greenland Ice'},
    'salar_uyuni': {'lat': -20.0, 'lon': -67.5, 'name': 'Salar de Uyuni'},
    'death_valley': {'lat': 36.5, 'lon': -117.0, 'name': 'Death Valley'},
}

TOTAL_REGIONS = len(REGIONS)


BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'ST_B10']
START_DATE = '2024-01-01'
END_DATE = '2024-06-30'
BUFFER_METERS = 15000


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
        .sort('CLOUD_COVER')
        .select(BANDS)
    )
    
    count = collection.size().getInfo()
    print(f"  Found {count} images")
    
    if count == 0:
        print(f"  No clear images available for {region_info['name']}")
        return None
    
    image = collection.first().clip(region)
    
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
    print(f"This may take several hours for {len(all_tasks)} band exports")
    print("="*60)
    
    while True:
        all_done = True
        running = 0
        completed = 0
        failed = 0
        
        for task_name, task in all_tasks:
            status = task.status()
            state = status['state']
            
            if state in ['READY', 'RUNNING']:
                all_done = False
                running += 1
            elif state == 'COMPLETED':
                completed += 1
            elif state == 'FAILED':
                failed += 1
        
        total = len(all_tasks)
        print(f"  Status: {completed}/{total} done, {running} running, {failed} failed")
        
        if all_done:
            print("\nAll tasks completed!")
            break
        
        time.sleep(60)  # Check every minute


def main():
    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception as exc:
        print(f"Failed to initialize Earth Engine: {exc}")
        print("Make sure you have authenticated with: earthengine authenticate")
        print(f"And that your project ID '{EE_PROJECT_ID}' is correct.")
        return

    print("Earth Engine initialized")
    print(f"Total regions configured: {TOTAL_REGIONS}")
    
    # Skip already downloaded regions
    existing = set()
    output_check = Path("data/landsat9/input")
    if output_check.exists():
        for d in output_check.iterdir():
            if d.is_dir():
                region_name = d.name.replace('_product', '')
                tif_count = len(list(d.glob("*.tif")))
                if tif_count == 4:
                    existing.add(region_name)
    
    print(f"Skipping {len(existing)} already downloaded regions")
    
    all_tasks = []
    skipped = 0
    
    for region_id, region_info in REGIONS.items():
        if region_id in existing:
            skipped += 1
            continue
        tasks = export_region(region_id, region_info)
        if tasks:
            all_tasks.extend(tasks)
    
    print(f"\nSkipped {skipped} existing regions")
    
    if not all_tasks:
        print("Nothing new to download - all regions already have data")
        return
    
    print(f"\nStarted {len(all_tasks)} export tasks total")
    print("\nIMPORTANT: Files will be saved to your Google Drive")
    print("Folder: InfraNova_Landsat9")
    print("\nMonitor at: https://code.earthengine.google.com/tasks")
    
    monitor_tasks(all_tasks)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Go to: https://drive.google.com")
    print("2. Find folder: InfraNova_Landsat9")
    print("3. Download new files")
    print("4. Run organize_files.py --source <downloaded Drive folder>")
    print("5. Run process_landsat_patches.py")


if __name__ == '__main__':
    main()
