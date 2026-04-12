import requests

payload = {
    "model": "qwen2.5-coder:3b",
    "messages": [
        {
            "role": "system",
            "content": "You are a Python coding assistant. Always respond with code in a ```python block."
        },
        {
            "role": "user",
            "content": (
                "Write a Python script that:\n"
                "1. Reads train.csv (columns: id, text, target)\n"
                "2. Trains TfidfVectorizer + LogisticRegression with 3-fold CV\n"
                "3. Prints mean F1 score\n"
                "4. Prints as the last line: METRICS: {\"f1\": 0.80, \"accuracy\": 0.80}\n\n"
                "Put ALL code inside a ```python code block."
            )
        }
    ],
    "stream": False,
    "temperature": 0.3,
}

r = requests.post("http://localhost:11434/v1/chat/completions", json=payload, timeout=300)
content = r.json()["choices"][0]["message"]["content"]
print("=== FULL RESPONSE ===")
print(content)
