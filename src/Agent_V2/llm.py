"""
llm.py — Ollama client.
Uses requests directly (no openai package needed).
Talks to Ollama's OpenAI-compatible endpoint at localhost:11434.
"""

import re
import requests
import json
import os
import time

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:latest"   # change to any model you have pulled
TIMEOUT = int(os.environ.get("DISASTER_AGENT_LLM_TIMEOUT", "600"))  # seconds to wait for LLM response


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not models:
                print("[LLM] WARNING: Ollama is running but no models are pulled.")
                print(f"[LLM] Run: ollama pull {self.model}")
            else:
                print(f"[LLM] Connected to Ollama. Available models: {models}")
                if self.model not in models and not any(self.model in m for m in models):
                    print(f"[LLM] WARNING: model '{self.model}' not found. Available: {models}")
                    print(f"[LLM] Run: ollama pull {self.model}")
        except requests.exceptions.ConnectionError:
            print("[LLM] ERROR: Cannot connect to Ollama at localhost:11434")
            print("[LLM] Make sure Ollama is running: start Ollama app or run 'ollama serve'")
            raise

    def _call(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        try:
            preview = user.strip().splitlines()[0][:90] if user.strip() else "(empty prompt)"
            started = time.perf_counter()
            print(f"[LLM] Request started | model={self.model} | timeout={TIMEOUT}s | prompt='{preview}'")
            r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            elapsed = time.perf_counter() - started
            print(f"[LLM] Request completed in {elapsed:.1f}s")
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            return "[LLM ERROR] Request timed out after {} seconds".format(TIMEOUT)
        except Exception as e:
            return f"[LLM ERROR] {e}"

    def propose(self, system: str, user: str) -> tuple[str, str]:
        """
        Ask the LLM to propose the next experiment.
        Returns (full_response, extracted_code).
        Returns ("", "") if LLM returned an error.
        """
        response = self._call(system, user)
        if response.startswith("[LLM ERROR]"):
            print(f"  [LLM] Error: {response}")
            return "", ""
        code = extract_code_block(response)
        return response, code

    def analyze(self, prompt: str) -> str:
        """Overrides base analyze to suppress error strings in memory."""
        result = self._call("You are a concise ML research analyst.", prompt)
        if result.startswith("[LLM ERROR]"):
            return "Analysis unavailable (LLM error)."
        return result



def extract_code_block(text: str) -> str:
    """
    Pull the first ```python ... ``` block out of the LLM response.
    Returns empty string if none found.
    """
    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try plain ``` block
    pattern2 = r"```\s*(.*?)```"
    match2 = re.search(pattern2, text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return ""
