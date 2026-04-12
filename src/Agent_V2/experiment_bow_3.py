"""Experiment-specific hooks for BoW_advanced_thr architecture."""

import re

from experiment_bow_2 import apply_light_autofixes as bow_adv_autofixes

ARCH_PROMPT = """
Architecture: BoW_advanced + OOF threshold tuning.

Same as BoW_advanced (multi-vectorizer TF-IDF ensemble with rank averaging), PLUS:
- NEVER reference train_idx/val_idx outside the CV loop.
- Define y = train_df['target'].values BEFORE any DRY_RUN slicing.

THRESHOLD TUNING (required):
- Store OOF probabilities: `oof_prob[val_idx] = ensemble_val` (do not threshold inside the loop).
- After CV, search for `best_thr` in [0.05..0.95] maximizing F1 on `(oof_prob >= thr)`.
- Set `oof_preds = (oof_prob >= best_thr).astype(int)` and print METRICS as the last line.
- Use the same `best_thr` to threshold `test_probs` for the submission.
"""

MODEL_PROMPT = (
    "Write ONLY a Python function `get_ensemble_probs(X_tr, y_tr, X_va, X_te)` "
    "that trains LogisticRegression, CalibratedClassifierCV(LinearSVC), and MultinomialNB "
    "on sparse matrix X_tr, combines with rank averaging, and returns "
    "{'valid': probs, 'test': probs}. Write ONLY the function, no imports.\n\n"
    "```python\ndef get_ensemble_probs(X_tr, y_tr, X_va, X_te):\n    ...\n```"
)


def preflight_issues(code: str, arch: str) -> list[str]:
    issues: list[str] = []
    has_oof_prob = bool(re.search(r"(?m)^\s*oof_prob[s]?\s*=\s*np\.zeros\(", code))
    writes_oof_prob = bool(re.search(r"oof_prob[s]?\[\s*(?:val|va)_(?:idx|index)\s*\]\s*=", code))

    if not (has_oof_prob and writes_oof_prob):
        issues.append("Missing OOF prob storage: define oof_prob and set oof_prob[val_idx] = ensemble_val inside the CV loop")
    if not re.search(r"best_thr|best_threshold|find_best_threshold|thresholds\s*=\s*np\.linspace", code):
        issues.append("Missing threshold tuning: select best_thr to maximize F1 on OOF probabilities")
    if not re.search(r"oof_preds\s*=\s*\(oof_prob[s]?\s*>?=", code):
        issues.append("Missing OOF thresholding: after selecting best_thr, compute oof_preds = (oof_prob(s) >= best_thr).astype(int) for METRICS")

    return issues


def apply_light_autofixes(code: str, arch: str) -> str:
    return bow_adv_autofixes(code, arch)


def build_repair_hint(stderr_text: str) -> str:
    if (
        "Missing threshold tuning" in stderr_text
        or "Missing OOF thresholding" in stderr_text
        or "Missing OOF prob storage" in stderr_text
    ):
        return (
            "BoW_advanced threshold tuning hint:\n"
            "Store OOF probabilities (not labels) inside each fold:\n"
            "  oof_prob[val_idx] = ensemble_val\n"
            "After CV, find best_thr maximizing F1 on (oof_prob >= thr), then:\n"
            "  oof_preds = (oof_prob >= best_thr).astype(int)\n"
            "Use best_thr to threshold test_probs for submission.\n\n"
        )
    return ""


def get_arch_prompt() -> str:
    return ARCH_PROMPT


def get_model_prompt() -> str:
    return MODEL_PROMPT


def get_template(arch_templates: dict[str, str]) -> str | None:
    return arch_templates.get("BoW_advanced_thr")
