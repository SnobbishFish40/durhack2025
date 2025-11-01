"""
    Configuration for API calls
"""

import os
from dotenv import load_dotenv

load_dotenv()

NASA_API_KEY = os.getenv('NASA_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not NASA_API_KEY or not ANTHROPIC_API_KEY:
    print("API keys not found!")

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
