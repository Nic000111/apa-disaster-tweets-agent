"""Experiment-specific hooks for CNN architecture."""

ARCH_PROMPT = """
Architecture: 1D Convolutional Neural Network on token sequences using PyTorch.

Tokenization: fit CountVectorizer(max_features=10000) on training texts.
For each tweet, take the indices of present words (up to 100), pad to length 100 with zeros.

Model structure:
- nn.Embedding(10001, 64, padding_idx=0)
- Two parallel nn.Conv1d branches: kernel_size=3 and kernel_size=5, each with 128 filters
- Global max pooling on each branch (torch.max over sequence dim)
- Concatenate both branches -> 256 dims
- nn.Linear(256, 1) output (single logit, no sigmoid in forward)

Training: Adam(lr=1e-3), BCEWithLogitsLoss, batch_size=64, 3 epochs (1 if dry-run).
Use torch.cuda if available.
OOF: sigmoid(logits) >= 0.5 for binary preds. Test: average sigmoid probs across folds.
"""

MODEL_PROMPT = (
    "Write ONLY a Python function `build_model(vocab_size)` returning a PyTorch nn.Module "
    "with Embedding + two Conv1d branches (kernel 3 and 5) + global max pool + Linear output. "
    "Single logit output, no sigmoid. torch and nn already imported. "
    "Write ONLY the function.\n\n"
    "```python\ndef build_model(vocab_size):\n    class CNN(nn.Module): ...\n    return CNN()\n```"
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
    return arch_templates.get("CNN")
