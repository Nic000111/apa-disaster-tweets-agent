# Disaster Tweets Autonomous Research Agent

## Overview

This project implements an autonomous machine learning agent for the Kaggle competition:
https://www.kaggle.com/competitions/nlp-getting-started

The system:
- proposes experiments
- trains models
- evaluates performance
- logs results
- iterates automatically

---

## Project Structure

apa-disaster-tweets-agent/
│
├── data/        (not included)
├── logs/
├── outputs/
├── models/
│
├── src/
│   ├── config.py
│   ├── agent.py
│   ├── advanced_agent.py
│   ├── dashboard.py
│   └── test_ollama.py
│
├── requirements.txt
├── .gitignore
└── README.md

---

## Setup

Clone:

git clone git@github.com:Nic000111/apa-disaster-tweets-agent.git
cd apa-disaster-tweets-agent

Create env:

python3 -m venv .venv
source .venv/bin/activate

Install:

pip install -r requirements.txt

---

## Data

Download Kaggle data:

mkdir -p data
kaggle competitions download -c nlp-getting-started -p data
unzip data/nlp-getting-started.zip -d data

---

## Run

Simple agent:

python src/agent.py

Advanced agent:

python src/advanced_agent.py

Dashboard:

streamlit run src/dashboard.py

---

## Config

Optional:

export OLLAMA_MODEL="gemma4:e4b"
export HF_DEFAULT_MODEL="distilroberta-base"
export DISASTER_AGENT_DATA_DIR="/your/path/to/data"

---

## Results

Simple model ≈ 0.76 F1  
Transformer ≈ 0.83 F1  

---

## Notes

- data is not included
- models/logs not committed
- works with any local LLM via Ollama

---

## Next Steps

- cross validation
- ensemble models
- submission pipeline

