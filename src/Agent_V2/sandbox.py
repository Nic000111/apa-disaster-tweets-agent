"""
sandbox.py — Safe code execution with dry-run sanity check.

Matches the diagram:
  dry run (1 epoch, 200 samples) → pass → full EXECUTE → MONITOR (poll log)
                                  → fail → capture error, skip full run

The generated code runs in a subprocess so crashes can't kill the agent.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

# ...existing code...
