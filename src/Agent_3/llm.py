"""
llm.py — Ollama client for Agent_3.
"""

from __future__ import annotations

import os
import re
import json
import threading
import time

import requests


OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "qwen2.5-coder:14b"
TIMEOUT = int(os.environ.get("DISASTER_AGENT_LLM_TIMEOUT", "1000"))
HEARTBEAT_SECS = int(os.environ.get("DISASTER_AGENT_LLM_HEARTBEAT_SECS", "10"))
STREAM = os.environ.get("DISASTER_AGENT_LLM_STREAM", "0") == "1"


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._check_connection()

    def _check_connection(self) -> None:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            if not models:
                print("[LLM] WARNING: Ollama is running but no models are pulled.")
            else:
                print(f"[LLM] Connected to Ollama. Available models: {models}")
                if self.model not in models and not any(self.model in m for m in models):
                    print(f"[LLM] WARNING: model '{self.model}' not found. Available: {models}")
        except requests.exceptions.ConnectionError:
            print("[LLM] ERROR: Cannot connect to Ollama at localhost:11434")
            raise

    def _heartbeat(self, stop: threading.Event, started: float) -> None:
        if HEARTBEAT_SECS <= 0:
            return
        while not stop.wait(HEARTBEAT_SECS):
            elapsed = time.perf_counter() - started
            print(f"[LLM] Still generating... {elapsed:.0f}s elapsed", flush=True)

    def _call_streaming(self, payload: dict, started: float) -> str:
        chunks: list[str] = []
        try:
            with requests.post(OLLAMA_URL, json=payload, timeout=(10, TIMEOUT), stream=True) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if line in ("[DONE]", "DONE"):
                        break
                    try:
                        data = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if not isinstance(data, dict):
                        continue
                    choice = (data.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content is None:
                        message = choice.get("message") or {}
                        content = message.get("content")
                    if not content:
                        continue
                    print(content, end="", flush=True)
                    chunks.append(content)
        finally:
            elapsed = time.perf_counter() - started
            if chunks:
                print("", flush=True)
            print(f"[LLM] Request completed in {elapsed:.1f}s", flush=True)
        return "".join(chunks)

    def _call(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "stream": STREAM,
        }
        try:
            preview = user.strip().splitlines()[0][:100] if user.strip() else "(empty prompt)"
            started = time.perf_counter()
            print(
                f"[LLM] Request started | model={self.model} | timeout={TIMEOUT}s | prompt='{preview}'",
                flush=True,
            )

            if STREAM:
                return self._call_streaming(payload, started)

            stop = threading.Event()
            t = threading.Thread(target=self._heartbeat, args=(stop, started), daemon=True)
            t.start()
            try:
                response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
                response.raise_for_status()
                elapsed = time.perf_counter() - started
                print(f"[LLM] Request completed in {elapsed:.1f}s", flush=True)
                return response.json()["choices"][0]["message"]["content"]
            finally:
                stop.set()
                t.join(timeout=1)
        except requests.exceptions.Timeout:
            return f"[LLM ERROR] Request timed out after {TIMEOUT} seconds"
        except Exception as exc:  # noqa: BLE001
            return f"[LLM ERROR] {exc}"

    def propose(self, system: str, user: str) -> tuple[str, str]:
        response = self._call(system, user)
        if response.startswith("[LLM ERROR]"):
            print(f"[LLM] Error: {response}")
            return "", ""
        return response, extract_code_block(response)

    def respond(self, system: str, user: str) -> str:
        response = self._call(system, user)
        if response.startswith("[LLM ERROR]"):
            print(f"[LLM] Error: {response}")
            return ""
        return response

    def analyze(self, prompt: str) -> str:
        response = self._call("You are a concise ML research analyst.", prompt)
        if response.startswith("[LLM ERROR]"):
            return "Analysis unavailable (LLM error)."
        return response


def extract_code_block(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""
