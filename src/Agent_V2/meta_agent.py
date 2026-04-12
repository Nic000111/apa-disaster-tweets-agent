"""
meta_agent.py — Meta-learning loop.

Flow per cycle:
  1. Run agent_fully_autonomous.py for N iterations
  2. Load experiment_log.json
  3. Feed full log to GPT-4.1 → get an improved standalone Python script
  4. Run that improved script directly via sandbox
  5. If F1 improves → save as best, submit to Kaggle (optional)
  6. Repeat for MAX_CYCLES

Usage:
  python meta_agent.py
  python meta_agent.py --cycles 5 --iter-per-cycle 3 --api-key sk-...
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ...existing code...
