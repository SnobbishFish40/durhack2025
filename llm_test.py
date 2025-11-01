import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
def request_llm_analysis():
    KEY = os.getenv('LLM_API_KEY')

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json"
    }

    data = {
        "model": "anthropic/claude-sonnet-4.5",
        "messages": [
        {"role": "user", "content": "Explain the impact of rising soil temperature on corn yield."}
        ],
    "max_tokens": 300
    }

    response = requests.post(url, headers=headers, json=data)

    return response.json()


print(request_llm_analysis())




