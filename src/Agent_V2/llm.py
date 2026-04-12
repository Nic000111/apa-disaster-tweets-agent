"""
llm.py — Ollama client.
Uses requests directly (no openai package needed).
Talks to Ollama's OpenAI-compatible endpoint at localhost:11434.
"""

import re
import requests
import json

# ...existing code...
