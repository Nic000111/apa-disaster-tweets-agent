import os
from pathlib import Path

# Root of the project repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------
# Data paths
# -----------------------------
# Default: use the repo's own data folder
# Teammates can override with:
# export DISASTER_AGENT_DATA_DIR="/full/path/to/data"
DATA_DIR = Path(
    os.getenv("DISASTER_AGENT_DATA_DIR", str(PROJECT_ROOT / "data"))
)

LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# -----------------------------
# Ollama settings
# -----------------------------
# Teammates can override with:
# export OLLAMA_URL="http://localhost:11434/api/generate"
# export OLLAMA_MODEL="gemma4:e4b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# -----------------------------
# Advanced transformer defaults
# -----------------------------
# Teammates can override with:
# export HF_DEFAULT_MODEL="distilroberta-base"
HF_DEFAULT_MODEL = os.getenv("HF_DEFAULT_MODEL", "distilroberta-base")

# -----------------------------
# Misc
# -----------------------------
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MAX_SIMPLE_EXPERIMENTS = int(os.getenv("MAX_SIMPLE_EXPERIMENTS", "3"))
MAX_ADVANCED_EXPERIMENTS = int(os.getenv("MAX_ADVANCED_EXPERIMENTS", "10"))

# Make sure output folders exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
