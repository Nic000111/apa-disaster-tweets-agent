"""
agent_fully_autonomous.py — Fully autonomous research agent.
This version asks the LLM to generate the entire experiment script for each iteration.
Enhanced with: forced arch sequencing, dry-run injection, METRICS guarantee.
"""

import argparse
import os
import re
import pandas as pd
import ast

from llm import OllamaClient
from memory import ExperimentMemory, REQUIRED_ARCHITECTURES
from sandbox import run_experiment, tail
from prompts import DATA_CONTEXT_TEMPLATE, ANALYSIS_PROMPT_TEMPLATE
from templates import (
    get_arch_prompt,
    SHARED_CONSTRAINTS,
    SHARED_REPAIR_CONSTRAINTS,
    ARCH_TEMPLATES,
    get_model_prompt,
    fill_template,
)

# ...existing code...
