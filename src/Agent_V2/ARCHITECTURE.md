# Autonomous ML Research Agent — Architecture

## Overview

An agent that autonomously designs, trains, evaluates, and iterates on ML models for the
[NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) Kaggle competition.
The agent uses a locally-hosted LLM (via Ollama) as its "brain" to drive the experimentation loop.

---

## File Structure

```
NEWW/
├── agent_fully_autonomous.py  # Entry point — fully autonomous loop
├── llm.py            # Ollama LLM client (THINK + REFLECT steps)
├── memory.py         # Tiered experiment memory (rolling log + milestones)
├── sandbox.py        # Safe code execution (dry run + full run + monitor)
├── templates.py      # Boilerplate-correct code templates per architecture
├── prompts.py        # All LLM prompt templates
│
├── train.csv         # Kaggle training data (7613 tweets)
├── test.csv          # Kaggle test data (3263 tweets)
├── experiment_log.json  # Persistent log of all experiments (auto-generated)
├── submissions/      # Kaggle submission CSVs (auto-generated)
│
├── disaster_tweets.ipynb  # Manual baseline (DeBERTa fine-tuning)
└── Untitled-1.py          # Original manual pipeline script
```

---

## Agent Loop

```
┌─────────────────────────────────────────────────────────────────┐
│              AGENT LOOP (agent_fully_autonomous.py)             │
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐  │
│  │  MEMORY  │───▶│   THINK   │───▶│ DRY RUN  │───▶│ EXECUTE │  │
│  │          │    │ (LLM #1)  │    │ 1 epoch  │    │ full run│  │
│  │ Tier 1:  │    │           │    │ 200 rows │    │  GPU    │  │
│  │ project  │    │ proposes  │    │          │    │         │  │
│  │ brief    │    │ arch +    │    │ fail ───▶│    │         │  │
│  │          │    │ model code│    │ skip     │    │         │  │
│  │ Tier 2:  │    └───────────┘    └──────────┘    └────┬────┘  │
│  │ rolling  │                                          │        │
│  │ log (15) │    ┌───────────┐    ┌──────────┐         │        │
│  │          │◀── │  UPDATE   │◀───│ REFLECT  │◀────────┘        │
│  │ Mileston:│    │  MEMORY   │    │ (LLM #2) │                  │
│  │ best kept│    │ + check   │    │ analyzes │                  │
│  └──────────┘    │ stopping  │    │ results  │                  │
│                  └───────────┘    └──────────┘                  │
│                                                                 │
│  Stopping criteria: F1 >= 0.88  OR  plateau (5 runs)  OR  N    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Sequence

The agent explores architectures in this order, inspired by the course content:

| # | Architecture   | Method                              | File           |
|---|----------------|-------------------------------------|----------------|
| 1 | `BoW`          | TF-IDF + Logistic Regression        | templates.py   |
| 2 | `BoW_advanced` | Multi-vectorizer + ensemble (LogReg + SVM + NB) with rank averaging | templates.py |
| 3 | `BoW_advanced_thr` | `BoW_advanced` + OOF threshold tuning | templates.py |
| 4 | `CNN`          | 1D CNN on token sequences (PyTorch) | templates.py   |
| 5 | `LSTM`         | BiLSTM (PyTorch, bidirectional)     | templates.py   |
| 6 | `Transformer`  | DistilBERT fine-tuning (HuggingFace)| LLM-generated  |

After all 6 are covered, the agent proposes improvements or ensembles based on what worked.

---

## Key Design Decisions

### Template-based code generation
For architectures 1–5, the LLM only writes a **single function** (`build_model()` or
`get_ensemble_probs()`). The surrounding boilerplate (data loading, CV loop, METRICS
printing, submission saving) is handled by guaranteed-correct templates in `templates.py`.
This prevents the recurring bugs (wrong column names, missing METRICS line, syntax errors)
that occur when small LLMs generate full scripts.

Each template also includes a **default fallback implementation** so experiments run even
if the LLM generates no code.

### Dry run gate (inspired by diagram)
Before every full training run, the sandbox runs the code with `AGENT_DRY_RUN=1`
(1 epoch, 200 samples, 60s timeout). If this fails, the full run is skipped and the
error is logged. This prevents wasting 10 minutes on broken code.

### Tiered memory (inspired by diagram)
- **Tier 1** — static project brief (always in prompt)
- **Tier 2** — rolling log of last 15 experiments (prevents repeating failures)
- **Milestones** — best F1 result kept forever

### Self-reflection (inspired by Dunivin et al., 2026)
The agent implements a two-stage refinement loop:
- **Stage 1 (BoW):** produces a simple, reliable baseline (TF-IDF + Logistic Regression)
- **Stage 2 (BoW_advanced):** adds richer features + an ensemble to improve over the baseline

This mirrors the paper's "precision-first, recall-oriented first pass + secondary LLM critic" design.

---

## Running the Agent

```bash
# Install Ollama and pull a model
ollama pull qwen2.5-coder:3b

# Run the agent
python agent_fully_autonomous.py --model qwen2.5-coder:3b --max-iter 7

# Run with stronger model (slower but better code)
python agent_fully_autonomous.py --model gemma4:latest --max-iter 7

# Start from scratch (ignore experiment_log.json) and run the first 3 steps:
# BoW -> BoW_advanced -> BoW_advanced_thr
python agent_fully_autonomous.py --fresh --max-iter 3
```

## Output Files

| File | Description |
|------|-------------|
| `experiment_log.json` | Full log of every experiment (prompt, code, metrics, analysis) |
| `submissions/*.csv` | Kaggle submission files, one per successful experiment |

---

## Manual Baseline

`disaster_tweets.ipynb` contains a manually tuned pipeline (TF-IDF ensemble + DeBERTa-v3-base
fine-tuning) that serves as the upper bound for comparison with the agent's results.
Expected F1: ~0.82–0.85 on the Kaggle leaderboard.
