import requests
from dotenv import load_dotenv
import os

load_dotenv()

KEY=os.getenv('NASA_API_KEY')

print("Testing connection to NASA Power endpoint!")

url = "https://power.larc.nasa.gov/api/temporal/daily/point"

params = {
    'parameters': 'T2M',  # temperature
    'community': 'AG',
    'longitude': -93.0977,
    'latitude': 41.8781,
    'start': '20250101',
    'end': '20251101',
    'format': 'JSON'
}


response = requests.get(url, params=params, timeout=30)

print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    data = response.json()

    print(data)

else:
    print(f"API returned status {response.status_code}")

