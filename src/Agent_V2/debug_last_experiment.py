"""
debug_last_experiment.py
Shows the full details of the most recent failed experiment:
- What code was generated
- What the dry run stderr was
- What the LLM analysis said
"""
import json

with open("experiment_log.json", encoding="utf-8") as f:
    exps = json.load(f)

# ...existing code...
