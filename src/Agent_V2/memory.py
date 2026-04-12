"""
memory.py — Tiered experiment memory (matches the diagram).

Tier 1 — Project brief: static context (set once, always included)
Tier 2 — Rolling log: last 15 experiment summaries (prevents repeats)
Milestones — Best results kept forever (never evicted)

Backed by experiment_log.json so the agent can resume across restarts.
"""

import json
import os
from datetime import datetime

# ...existing code...
