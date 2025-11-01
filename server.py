import requests
from datetime import datetime, timedelta
import os
import pandas as pd
from anthropic import Anthropic

load_dotenv()
NASA_KEY = os.getenv('NASA_API_KEY')
CLAUDE_KEY = os.getenv('ANTHROPIC_API_KEY')

def get_nasa_csv(lat, lon, startDate, endDate):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": ','.join([
            'T2M',              # Temperature
            'T2M_MAX',          # Max temp
            'T2M_MIN',          # Min temp
            'PRECTOTCORR',      # Precipitation
            'RH2M',             # Humidity
            'GWETPROF',         # Surface soil wetness ⭐
            'GWETROOT',         # Root zone soil wetness ⭐
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
    
    data = StringIO(response.text)
    pureData = pd.read_csv(data, skiprows=13) # Skip header

    return pureData


def request_llm_analysis(stats, crop_type):
    client = Anthropic(api_key=CLAUDE_KEY)

    response = client.messages.create(
        model="claude-sonnet-4-5-2025-09-29",
        messages=[
            {"role": "user", "content": f"Here is a CSV file full of statistical predictions about conditions relating to agricultural success in a region :\n{stats}\nPlease break down clearly what implications this may have on {crop_type} growth in the area."}
        ]
    )
    return response.content[0].text)
