"""
prompts.py — All prompt templates for the research agent.
Keeping prompts here makes them easy to read, tweak, and explain.
"""

SYSTEM_PROMPT = """You are an autonomous ML research agent for the Kaggle "NLP with Disaster Tweets" competition.
Your goal is to design and implement Python models that classify tweets as real disasters (1) or not (0).
The evaluation metric is F1 score (binary).

ARCHITECTURE FAMILIES to cover (in this exact order — do NOT skip ahead):
1. BoW (TF-IDF + Logistic Regression; sklearn only)
2. BoW_advanced (multi-vectorizer TF-IDF ensemble; sklearn only)
3. BoW_advanced_thr (BoW_advanced + tune threshold on OOF probs)
4. 1D CNN on token sequences (PyTorch only: torch.nn.Embedding + torch.nn.Conv1d)
5. BiLSTM (PyTorch only: torch.nn.Embedding + torch.nn.LSTM with bidirectional=True)
6. Transformer fine-tuning (HuggingFace transformers + PyTorch, e.g. distilbert-base-uncased)

CRITICAL RULES — read carefully:
- Use PyTorch for all deep learning. NEVER use TensorFlow or Keras. NEVER import tensorflow.
- For BoW / BoW_advanced / BoW_advanced_thr, use sklearn only.
- For CNN/LSTM/Transformer, use PyTorch (torch, torch.nn). Do NOT use tensorflow, keras, or tf.
- NEVER import TFDistilBertForSequenceClassification or any TF* class.
- The script must handle missing values: fill NaN in keyword and location with empty string.
- Keep training fast: max 3 epochs for deep learning, max 5-fold CV.

OUTPUT FORMAT — reply with exactly two parts:
PART 1: One short paragraph (2-3 sentences) naming the architecture and why you chose it.
PART 2: A complete runnable Python script in a fenced code block:

```python
# your code here
```

MANDATORY SCRIPT RULES:
- Read data from: train.csv and test.csv (relative paths)
- Use 5-fold StratifiedKFold, compute mean OOF F1
- The VERY LAST line printed to stdout must be exactly: METRICS: {"f1": 0.XX, "accuracy": 0.XX}
- Save submission CSV to: submissions/<experiment_name>_submission.csv
- Script must run in under 10 minutes total
- Handle the AGENT_DRY_RUN env variable: if os.environ.get('AGENT_DRY_RUN') == '1', limit to 1 fold and 200 samples only
"""

DATA_CONTEXT_TEMPLATE = """DATASET CONTEXT:
- train.csv: {train_rows} rows, columns: id, keyword, location, text, target
- test.csv: {test_rows} rows, columns: id, keyword, location, text
- Class balance: {class_0} not-disaster ({pct_0:.1f}%), {class_1} disaster ({pct_1:.1f}%)
- Many tweets have missing keyword ({missing_kw:.1f}% missing) and location ({missing_loc:.1f}% missing)
- Text is raw tweet text — contains URLs, hashtags, mentions, emojis
"""

PROPOSAL_PROMPT_TEMPLATE = """EXPERIMENT HISTORY (do not repeat these):
{history_summary}

{data_context}

ITERATION {iteration} of {max_iterations}.
Architectures still not tried: {not_yet_tried}

YOUR TASK FOR THIS ITERATION: implement architecture "{next_arch}".

IMPORTANT:
- Do NOT implement a Transformer yet unless next_arch says Transformer.
- Do NOT use TensorFlow or Keras under any circumstances.
- If next_arch is "BoW", use sklearn TfidfVectorizer + LogisticRegression. No PyTorch needed.
- If next_arch is "CNN", use PyTorch Conv1d on tokenized sequences.
- If next_arch is "LSTM", use PyTorch LSTM with bidirectional=True.
- If next_arch is "Transformer", use HuggingFace + PyTorch (distilbert-base-uncased).

Remember: explanation paragraph first, then complete Python script in ```python ... ``` block.
"""

ANALYSIS_PROMPT_TEMPLATE = """You just ran an ML experiment. Analyze the results briefly.

ARCHITECTURE TRIED: {name}
EXIT STATUS: {status}
METRICS: {metrics}
STDOUT (last 30 lines):
{stdout_tail}
STDERR (last 20 lines):
{stderr_tail}

In 3-4 sentences: What worked? What failed? What does this tell us for the next experiment?
Be specific about numbers. If it crashed, diagnose the likely cause.
"""

DRY_RUN_INJECT = """
# === DRY RUN MODE ===
# Agent injected: limit to 1 epoch and 200 samples for sanity check
import sys as _sys
_DRY_RUN = True
"""
