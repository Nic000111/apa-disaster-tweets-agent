# Disaster Tweets Autonomous Research Agent

This repository contains a team project for the **Advanced Predictive Analytics 2025/2026** course.  
The goal is to build an **AI-powered autonomous research agent** for the Kaggle competition **NLP with Disaster Tweets**.

The project currently includes:

- a **simple agent** that uses a local LLM through Ollama to propose lightweight text-classification experiments
- an **advanced agent** that runs transformer-based experiments for stronger performance
- a **dashboard** to compare experiments across all agent versions
- a shared **configuration system** so teammates can run the project with their own local paths and LLM models

---

## 1. Project Goal

The course project requires an autonomous system that can:

1. load and explore training data
2. propose a model or experiment configuration
3. run training automatically
4. evaluate the model
5. log the results
6. iterate and improve over time

This repository implements that idea in two stages:

### Simple agent
A lightweight proof-of-concept agent that:
- uses Ollama as the local LLM
- asks the LLM which experiment to try next
- trains simple neural text models such as embedding-average models or LSTMs
- logs results and keeps a history of past experiments

### Advanced agent
A more competition-grade pipeline that:
- uses pretrained transformers
- runs a structured set of stronger experiments
- tunes the prediction threshold for F1
- logs and compares multiple experiment configurations

---

## 2. Repository Structure

```text
disaster_tweets_agent/
│
├── data/                  # Kaggle files (not committed)
├── logs/                  # Experiment logs
├── outputs/               # Best results, submission files
├── models/                # Saved transformer checkpoints
├── src/
│   ├── config.py          # Centralized config
│   ├── agent.py           # Simple LLM-driven agent
│   ├── advanced_agent.py  # Transformer-based experiment runner
│   ├── dashboard.py       # Streamlit dashboard
│   └── test_ollama.py     # Simple connection test for Ollama
├── .gitignore
├── requirements.txt
└── README.md

