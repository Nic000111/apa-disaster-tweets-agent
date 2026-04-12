"""Experiment-specific hooks for Transformer architecture."""

ARCH_PROMPT = """
Architecture: Fine-tuned DistilBERT using HuggingFace Transformers + PyTorch.

Use AutoTokenizer and AutoModelForSequenceClassification from 'distilbert-base-uncased'.
Input format: keyword + ' [SEP] ' + text (or just text if keyword is empty).
Tokenize with max_length=128, truncation=True, padding='max_length'.

Training via HuggingFace Trainer:
- 2 epochs, batch_size=16 (train) / 32 (eval)
- learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1
- Use compute_metrics returning {'f1': f1_score(labels, preds)}
- fp16=False, report_to='none'

After each fold: collect logits from trainer.predict(), apply softmax to get class probs.
OOF: argmax for binary preds. Test: average class-1 probs across folds.
"""


def preflight_issues(code: str, arch: str) -> list[str]:
    return []


def apply_light_autofixes(code: str, arch: str) -> str:
    return code


def build_repair_hint(stderr_text: str) -> str:
    return ""


def get_arch_prompt() -> str:
    return ARCH_PROMPT


def get_model_prompt() -> str:
    return ""


def get_template(arch_templates: dict[str, str]) -> str | None:
    return None
