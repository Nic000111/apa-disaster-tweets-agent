"""Experiment-specific hooks for LSTM architecture."""

ARCH_PROMPT = """
Architecture: Bidirectional LSTM on token sequences using PyTorch.

Tokenization: fit CountVectorizer(max_features=10000) on training texts.
For each tweet, take the indices of present words (up to 100), pad to length 100 with zeros.

Model structure:
- nn.Embedding(10001, 64, padding_idx=0)
- nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
- Take the last hidden state from both directions: torch.cat([h[-2], h[-1]], dim=1) -> 256 dims
- nn.Linear(256, 1) output (single logit, no sigmoid in forward)

Training: Adam(lr=1e-3), BCEWithLogitsLoss, batch_size=64, 3 epochs (1 if dry-run).
Use torch.cuda if available.
OOF: sigmoid(logits) >= 0.5 for binary preds. Test: average sigmoid probs across folds.
"""

MODEL_PROMPT = (
    "Write ONLY a Python function `build_model(vocab_size)` returning a PyTorch nn.Module "
    "with Embedding + BiLSTM(hidden=128, bidirectional=True) + last hidden state concat + Linear output. "
    "Single logit output, no sigmoid. torch and nn already imported. "
    "Write ONLY the function.\n\n"
    "```python\ndef build_model(vocab_size):\n    class BiLSTM(nn.Module): ...\n    return BiLSTM()\n```"
)


def preflight_issues(code: str, arch: str) -> list[str]:
    issues: list[str] = []
    if "import torch" not in code:
        issues.append("Missing import: import torch")
    if "from torch.utils.data import DataLoader, TensorDataset" not in code:
        issues.append("Missing import: from torch.utils.data import DataLoader, TensorDataset")
    if "CountVectorizer" not in code:
        issues.append("Missing sequence encoding with CountVectorizer(max_features=10000)")
    if "model.predict_proba(" in code:
        issues.append("Invalid PyTorch call: use torch.sigmoid(model(...)) instead of model.predict_proba(...)")
    if "tokenizer.tokenize(" in code or "convert_tokens_to_ids(" in code:
        issues.append("Invalid tokenizer API for this setup: use CountVectorizer-based integer sequence encoding")
    return issues


def apply_light_autofixes(code: str, arch: str) -> str:
    return code


def build_repair_hint(stderr_text: str) -> str:
    return ""


def get_arch_prompt() -> str:
    return ARCH_PROMPT


def get_model_prompt() -> str:
    return MODEL_PROMPT


def get_template(arch_templates: dict[str, str]) -> str | None:
    return arch_templates.get("LSTM")
