"""
export_experiments.py
Saves each experiment's generated code to a .py file in experiments/ folder.
Also prints a summary table.
"""
import json
import os

with open("experiment_log.json", encoding="utf-8") as f:
    exps = json.load(f)

# ...existing code...
