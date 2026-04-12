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
AGENT_V2_DIR = PROJECT_ROOT / "src" / "Agent_V2"
AGENT_V2_EXPERIMENT_LOG_PATH = AGENT_V2_DIR / "experiment_log.json"

# -----------------------------
# Agent log/output filenames
# -----------------------------
# New, versioned filenames (preferred)
V1_SIMPLE_LOG_PATH = LOGS_DIR / "agent_v1_simple.jsonl"
V2_TRANSFORMER_LOG_PATH = LOGS_DIR / "agent_v2_transformer.jsonl"

V1_SIMPLE_BEST_PATH = OUTPUTS_DIR / "agent_v1_simple_best.json"
V2_TRANSFORMER_BEST_PATH = OUTPUTS_DIR / "agent_v2_transformer_best.json"

# Backwards-compatible legacy filenames (read-only)
LEGACY_SIMPLE_LOG_PATH = LOGS_DIR / "experiments.jsonl"
LEGACY_ADVANCED_LOG_PATH = LOGS_DIR / "advanced_experiments.jsonl"
LEGACY_SIMPLE_BEST_PATH = OUTPUTS_DIR / "best_result.json"
LEGACY_ADVANCED_BEST_PATH = OUTPUTS_DIR / "advanced_best_result.json"

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

# Clearer aliases used by `src/agent_vs.py` and `src/agents/*`.
MAX_V1_EXPERIMENTS = MAX_SIMPLE_EXPERIMENTS
MAX_V2_EXPERIMENTS = MAX_ADVANCED_EXPERIMENTS

# Make sure output folders exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
