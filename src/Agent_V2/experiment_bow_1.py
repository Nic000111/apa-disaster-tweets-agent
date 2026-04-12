"""Experiment-specific hooks for BoW architecture."""

ARCH_PROMPT = """
Architecture: Simple TF-IDF + Logistic Regression pipeline.

Required imports (include ALL of these):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

Use sklearn Pipeline with:
- TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2, strip_accents='unicode')
- LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='liblinear')

Apply to the 'text' column only. Use predict() for OOF labels, predict_proba()[:,1] for test averaging.
"""

MODEL_PROMPT = (
        "Write ONLY a Python function `build_model()` returning a sklearn Pipeline with "
        "TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2) and "
        "LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000). "
        "Write ONLY the function, no imports.\n\n"
        "```python\ndef build_model():\n    return make_pipeline(...)\n```"
)


def preflight_issues(code: str, arch: str) -> list[str]:
    return []


def apply_light_autofixes(code: str, arch: str) -> str:
    return code


def build_repair_hint(stderr_text: str) -> str:
    return ""


def get_arch_prompt() -> str:
    return ARCH_PROMPT


def get_model_prompt() -> str:
    return MODEL_PROMPT


def get_template(arch_templates: dict[str, str]) -> str | None:
    return arch_templates.get("BoW")
