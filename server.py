import requests
from datetime import datetime, timedelta
import os

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
