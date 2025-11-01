import requests
from datetime import datetime, timedelta
import os

load_dotenv()
NASA_KEY = os.getenv('NASA_API_KEY')
CLAUDE_KEY = os.getenv('ANTHROPIC_API_KEY')

def get_nasa_csv(lat, lon, startDate, endDate):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
    }
