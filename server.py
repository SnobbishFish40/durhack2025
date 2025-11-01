import requests
import os
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
NASA_KEY = os.getenv('NASA_API_KEY')
LLM_KEY = os.getenv('LLM_API_KEY')

def get_nasa_csv(lat, lon, startDate, endDate):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": ','.join([
            'T2M',              # Temperature
            'T2M_MAX',          # Max temp
            'T2M_MIN',          # Min temp
            'PRECTOTCORR',      # Precipitation
            'RH2M',             # Humidity
            'GWETPROF',         # Surface soil wetness
            'GWETROOT',         # Root zone soil wetness
            'ALLSKY_SFC_PAR_TOT', # Light for photosynthesis
            'WS2M'              # Wind speed
        ]),
        'community': 'AG',
        'longitude': lon,
        'latitude': lat,
        'start': startDate,
        'end': endDate,
        'format': 'CSV'
    }
    
    # response = requests.get(url, params=params, timeout=30)
    # response.raise_for_status()
    
    # data = StringIO(response.text)
    # pureData = pd.read_csv(data, skiprows=13) # Skip header

    return "date, temp, ndvi\n2020-01-01,12.3,0.41\n2020-01-08,11.7,0.44"
    return pureData


def request_llm_analysis(stats, crop_type):
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "anthropic/claude-sonnet-4.5",
        "message": [
            {"role": "user", "content": f"Attached is some data and predictions based on that data about conditions in a specific region in which someone wants to grow some {crop}. Please analyse the predictions and explain what the implications of these future conditions could be. {stats}"}
        ],
        "max_tokens": 300
    }
 
    response = requests.post(url, headers=headers, json=data)

    return response.json()
