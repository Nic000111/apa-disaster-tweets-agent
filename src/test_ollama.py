import requests

url = "http://localhost:11434/api/generate"

data = {
    "model": "gemma4:e4b",
    "prompt": "Say only: connection works",
    "stream": False
}

response = requests.post(url, json=data, timeout=120)
response.raise_for_status()

print(response.json()["response"])
