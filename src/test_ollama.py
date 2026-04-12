import requests

from config import OLLAMA_MODEL, OLLAMA_URL

data = {
    "model": OLLAMA_MODEL,
    "prompt": "Say only: connection works",
    "stream": False
}

response = requests.post(OLLAMA_URL, json=data, timeout=120)
response.raise_for_status()

print(response.json()["response"])
