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
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    return response.text

def request_llm_analysis(stats, crop, lat, lon):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "anthropic/claude-sonnet-4.5",
        "messages": [
            {"role": "user", "content": f"Attached is some data and predictions based on that data about conditions in a specific region (DO NOT PUT '[SPECIFIED REGION]' into this it looks tacky! and not professional) in which someone wants to grow some {crop}. You are a agricultural analyst, you must be able to provide this farmer with both analysis and actionable recommendations. Please analyse the predictions and explain in extreme detail what the implications of these future conditions could be - i.e a full professional report. {stats}"}
        ],
        "max_tokens": 20000
    }
 
    response = requests.post(url, headers=headers, json=data)

    return response.json()

