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

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_ITERATIONS = 2
TARGET_F1 = 0.88
PLATEAU_WINDOW = 5
MIN_IMPROVEMENT = 0.002
MAX_REPAIR_ATTEMPTS = 2
DATA_DIR_ENV = "DISASTER_AGENT_DATA_DIR"
DEFAULT_DATA_DIR = "data"

# Fully-autonomous exploration order (includes the tuned BoW_advanced variant as a separate step).
ARCH_SEQUENCE = ["BoW", "BoW_advanced", "BoW_advanced_thr", "CNN", "LSTM", "Transformer"]

FULL_SYSTEM = (
    "You are an expert ML engineer writing Python scripts for the Kaggle Disaster Tweets competition.\n"
    "Respond with ONLY a ```python code block. No text outside the code block.\n\n"
    + SHARED_CONSTRAINTS
)
REPAIR_SYSTEM = (
    "You are an expert Python code repair assistant for ML scripts.\n"
    "Respond with ONLY a ```python code block. No text outside the code block.\n\n"
    + SHARED_REPAIR_CONSTRAINTS
)


def get_data_paths() -> tuple[str, str]:
    """Resolve train/test CSV paths from env-configured data directory, with root fallback."""
    data_dir = os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path

    # Backward-compatibility for older setups that kept files at repo root.
    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        print("[Data] WARNING: using train.csv/test.csv from repo root; prefer data/train.csv and data/test.csv")
        return "train.csv", "test.csv"

    raise FileNotFoundError(
        "Could not find dataset files. Expected either "
        f"'{train_path}' and '{test_path}', or repo-root train.csv/test.csv."
    )


def build_data_context() -> str:
    train_path, test_path = get_data_paths()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    vc = train["target"].value_counts()
    total = len(train)
    return DATA_CONTEXT_TEMPLATE.format(
        train_rows=len(train),
        test_rows=len(test),
        class_0=vc.get(0, 0),
        class_1=vc.get(1, 0),
        pct_0=100 * vc.get(0, 0) / total,
        pct_1=100 * vc.get(1, 0) / total,
        missing_kw=100 * train["keyword"].isna().mean(),
        missing_loc=100 * train["location"].isna().mean(),
    )


def infer_architecture(text: str) -> str:
    text_lower = text.lower()
    if "bow_advanced_thr" in text_lower or "bowadvanced2" in text_lower or "threshold tuning" in text_lower:
        return "BoW_advanced_thr"
    if any(w in text_lower for w in ["transformer", "bert", "distilbert", "deberta"]):
        return "Transformer"
    if any(w in text_lower for w in ["lstm", "gru", "bilstm", "birnn"]):
        return "LSTM"
    if any(w in text_lower for w in ["conv1d", "cnn", "convolutional"]):
        return "CNN"
    if "ensemble" in text_lower or "bow_advanced" in text_lower:
        return "BoW_advanced"
    return "BoW"


def syntax_check(code: str) -> tuple[bool, str]:
    """Check Python syntax before running. Returns (ok, error_message)."""
    import ast
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


def preflight_issues(code: str, arch: str) -> list[str]:
    """Detect common autonomous-generation failures before execution."""
    issues: list[str] = []

    # Reject prompt/requirements leakage into the code body.
    if re.search(r"(?m)^(MANDATORY_SCRIPT_REQUIREMENTS|MANDATORY SCRIPT REQUIREMENTS|DATA LOADING|CROSS-VALIDATION|SUBMISSION|METRICS)\s*:\s*$", code):
        issues.append("Prompt leakage: remove headings like 'DATA LOADING:' / 'CROSS-VALIDATION:' / 'SUBMISSION:' / 'METRICS:' from the script (code only)")

    # Common failure: wrapping runtime flags into a dict and then referencing undefined DRY_RUN/N_SPLITS.
    if re.search(r"(?m)^\s*RUNTIME_FLAGS\s*=\s*{", code):
        issues.append("Forbidden runtime flags: define DRY_RUN and N_SPLITS as plain variables (no RUNTIME_FLAGS dict)")

    if "hstack(" in code and "from scipy.sparse import hstack" not in code:
        issues.append("Missing import: from scipy.sparse import hstack")

    if re.search(r"\brankdata\(", code) and "from scipy.stats import rankdata" not in code:
        issues.append("Missing import: from scipy.stats import rankdata")

    if arch in ("BoW_advanced", "BoW_advanced_thr"):
        # Enforce slicing from pre-built .values arrays, not pandas Series with label-based indexing.
        if re.search(r"train_df\[['\"]keyword['\"]\]\s*\[\s*train_idx\s*\]", code) or re.search(r"train_df\[['\"]keyword['\"]\]\s*\[\s*val_idx\s*\]", code):
            issues.append("Unsafe keyword slicing: build X_kw = train_df['keyword'].values before CV and slice X_kw[train_idx]/X_kw[val_idx] (avoid train_df['keyword'][train_idx])")
        if re.search(r"train_df\[['\"]location['\"]\]\s*\[\s*train_idx\s*\]", code) or re.search(r"train_df\[['\"]location['\"]\]\s*\[\s*val_idx\s*\]", code):
            issues.append("Unsafe location slicing: build X_loc = train_df['location'].values before CV and slice X_loc[train_idx]/X_loc[val_idx] (avoid train_df['location'][train_idx])")

        if "import scipy.sparse.hstack" in code:
            issues.append("Invalid import: use 'from scipy.sparse import hstack' (never 'import scipy.sparse.hstack')")
        if "from sklearn.calibrated import CalibratedClassifierCV" in code:
            issues.append("Invalid import: use 'from sklearn.calibration import CalibratedClassifierCV' (no sklearn.calibrated module)")
        if "from sklearn.calibrated_classifier_cv import CalibratedClassifierCV" in code:
            issues.append("Invalid import: use 'from sklearn.calibration import CalibratedClassifierCV'")
        if re.search(r"(?m)^\s*y\s*=\s*train_df\[['\"]target['\"]\]\s*$", code):
            issues.append("Bare y Series: define y = train_df['target'].values (Series without .values breaks rules and can confuse CV)")
        if "skf.split(X_text, train_df['target'])" in code or 'skf.split(X_text, train_df["target"])' in code:
            issues.append("Bare target in CV: use y = train_df['target'].values then skf.split(X_text, y)")
        if re.search(r"(?s)for\s+fold\s*,\s*\(train_idx,\s*val_idx\)\s+in\s+enumerate\([^\)]*skf\.split[^\)]*\)\s*:\s*.*?\n\s*if\s+DRY_RUN\s*:", code):
            issues.append("DRY_RUN misused: slice to 200 rows BEFORE the CV loop; do not branch on DRY_RUN inside the fold loop")

        # Index variables used before loop (common LLM bug).
        loop_pos = code.find("for fold")
        if loop_pos != -1:
            pre_loop = code[:loop_pos]
            if re.search(r"\btrain_idx\b", pre_loop) or re.search(r"\bval_idx\b", pre_loop):
                issues.append("Index variables used before CV loop: never reference train_idx/val_idx outside the CV loop")

        # y sliced before being defined (common DRY_RUN slip).
        if "y = y[:200]" in code and not re.search(r"(?m)^\s*y\s*=\s*train_df\[['\"]target['\"]\]\.values", code):
            issues.append("y sliced before definition: define y = train_df['target'].values before DRY_RUN slicing")

        # Column usage sanity: keyword/location vectorizers must not be fit on tweet text.
        if ("kw_vec" in code or "keyword" in code) and "train_df['keyword']" not in code and 'train_df["keyword"]' not in code:
            if "kw_vec" in code:
                issues.append("Keyword features missing: define X_kw = train_df['keyword'].values and fit kw_vec on X_tr_kw, not on X_tr text")
        if ("loc_vec" in code or "location" in code) and "train_df['location']" not in code and 'train_df["location"]' not in code:
            if "loc_vec" in code:
                issues.append("Location features missing: define X_loc = train_df['location'].values and fit loc_vec on X_tr_loc, not on X_tr text")

        # Common variable-overwrite bug: X_va assigned the *train* stacked matrix.
        if re.search(r"\bX_va\s*=\s*hstack\(\[word_tr", code):
            issues.append("Validation matrix bug: do not assign X_va = hstack([word_tr,...]); build X_tr_sparse from *_tr and X_va_sparse from *_va")
        if re.search(r"\bX_tr\s*=\s*hstack\(\[word_va", code):
            issues.append("Train matrix bug: do not build X_tr from validation features; create X_tr_sparse from *_tr")
        if re.search(r"(?m)^\s*X_va_sparse\s*=\s*hstack\(\[[^\]]*_tr_sparse", code):
            issues.append("Validation matrix bug: X_va_sparse is built from *_tr_sparse; build it from vectorizer.transform(X_va_*) outputs")
        if re.search(r"(?m)^\s*clf\d\.fit\(\s*X_tr_sparse", code) and not re.search(r"(?m)^\s*X_tr_sparse\s*=", code):
            issues.append("Undefined X_tr_sparse: build X_tr_sparse via hstack([...]).tocsr() before fitting classifiers")
        if re.search(r"\bX_tr_text\s*=\s*X_text\[", code) and not re.search(r"(?m)^\s*X_text\s*=", code):
            issues.append("Undefined X_text: define X_text = train_df['text'].values before CV (or use X directly)")

        # Training on raw text instead of sparse features (only flag when X_tr comes from X[...] slicing).
        if re.search(r"\bX_tr\s*,\s*X_va\s*=\s*X\[\s*train_idx\s*\]\s*,\s*X\[\s*val_idx\s*\]", code):
            if (
                re.search(r"\bclf1\.fit\(\s*X_tr\s*,\s*y_tr\s*\)", code)
                or re.search(r"\bclf2\.fit\(\s*X_tr\s*,\s*y_tr\s*\)", code)
                or re.search(r"\bclf3\.fit\(\s*X_tr\s*,\s*y_tr\s*\)", code)
            ):
                issues.append("Model fit bug: classifiers must be fit on the stacked sparse matrix (e.g., X_tr_sparse), not raw X_tr text")

        # Mis-averaging test probabilities: summing 3 model probs without rank-averaging.
        if code.count("test_probs +=") >= 2 and "ensemble_test" not in code:
            issues.append("Test aggregation bug: compute ensemble_test via rank averaging and do test_probs += ensemble_test / N_SPLITS (don’t sum 3 model probs)")
        if "test_probs +=" in code and "ensemble_test" not in code and re.search(r"test_probs\s*\+=.*predict_proba\(", code):
            issues.append("Test aggregation bug: compute ensemble_test via rank averaging and do test_probs += ensemble_test / N_SPLITS (don’t sum model probs directly)")

        # Validation aggregation should also use rank averaging (not plain mean of probs).
        if re.search(r"oof_preds\[\s*val_idx\s*\]\s*=\s*\(.*predict_proba\(", code, flags=re.S):
            if re.search(r"oof_preds\[\s*val_idx\s*\]\s*=\s*\(.*?\)\s*/\s*3", code, flags=re.S):
                issues.append("Validation aggregation bug: use rank averaging (rankdata + mean) for validation ensemble, not plain mean of probabilities")

        # LinearSVC does not support predict_proba unless calibrated.
        if "LinearSVC" in code and "predict_proba" in code and "CalibratedClassifierCV(" not in code:
            issues.append("Probability bug: LinearSVC has no predict_proba; wrap it with CalibratedClassifierCV(estimator=Pipeline([...LinearSVC...]), cv=2)")

        # Missing or misplaced fold-wise test accumulation.
        if re.search(r"(?m)^test_probs\s*\+\=", code):
            issues.append("Incorrect test accumulation location: update test_probs inside the CV loop (indented), not after the loop")
        if ("X_te_sparse" in code or "X_te =" in code) and not re.search(r"predict_proba\(\s*X_te", code) and "ensemble_test" not in code:
            issues.append("Missing test prediction: compute proba*_test on X_te(_sparse), rank-average to ensemble_test, and accumulate into test_probs inside each fold")
        if re.search(r"test_probs\s*=\s*np\.zeros\(\s*len\(\s*X_test\s*\)\s*\)", code) and not re.search(r"(?m)^\s*X_test\s*=", code):
            issues.append("Undefined X_test: use X_test_text or define X_test = test_df['text'].values before using len(X_test)")
        if re.search(r"\bskf\.split\(\s*X_text\s*,\s*y\s*\)", code) and not re.search(r"(?m)^\s*y\s*=", code):
            issues.append("Missing target y: define y = train_df['target'].values before CV")

        # Undefined test sparse matrix (common name mix-up with X_te_sparse).
        if "X_test_sparse" in code and not re.search(r"(?m)^\s*X_test_sparse\s*=", code):
            issues.append("Undefined X_test_sparse: build X_te_sparse (or X_test_sparse) via vectorizers inside each fold before predict_proba")
        if re.search(r"predict_proba\(\s*X_te_sparse\s*\)", code) and not re.search(r"(?m)^\s*X_te_sparse\s*=", code):
            issues.append("Undefined X_te_sparse: build X_te_sparse inside each fold via hstack([...]).tocsr() before predict_proba(X_te_sparse)")

        # Missing validation / OOF predictions (common: only computing test_probs).
        # Accept both `oof_prob` and `oof_probs` naming (templates use `oof_probs`).
        has_oof_prob = bool(re.search(r"(?m)^\s*oof_prob[s]?\s*=\s*np\.zeros\(", code))
        writes_oof_prob = bool(re.search(r"oof_prob[s]?\[\s*(?:val|va)_(?:idx|index)\s*\]\s*=", code))
        writes_oof_preds = bool(re.search(r"oof_preds\[\s*(?:val|va)_(?:idx|index)\s*\]\s*=", code))
        if (("oof_preds" in code) or ("oof_prob" in code)) and not (writes_oof_preds or (has_oof_prob and writes_oof_prob)):
            issues.append("Missing OOF update: set either oof_preds[val_idx] (labels) or oof_prob(s)[val_idx] (probabilities) inside the CV loop")
        if "X_va_sparse" in code and not re.search(r"predict_proba\(\s*X_va_sparse\s*\)", code):
            issues.append("Missing validation prediction: compute proba1/proba2/proba3 on X_va_sparse and build ensemble for oof_preds")
        if re.search(r"\bproba\d+_test\s*=", code) and not re.search(r"predict_proba\(\s*X_va", code):
            issues.append("Validation missing: you compute proba*_test but never predict on validation (X_va*); add validation probs for OOF metrics")
        if re.search(r"oof_preds\[\s*val_idx\s*\][^\n]*ensemble_test", code):
            issues.append("OOF bug: do not use ensemble_test (test) for oof_preds; compute validation ensemble from X_va_sparse")
        if arch == "BoW_advanced_thr":
            if not (has_oof_prob and writes_oof_prob):
                issues.append("Missing OOF prob storage: define oof_prob and set oof_prob[val_idx] = ensemble_val inside the CV loop")
            if not re.search(r"best_thr|best_threshold|find_best_threshold|thresholds\s*=\s*np\.linspace", code):
                issues.append("Missing threshold tuning: select best_thr to maximize F1 on OOF probabilities")
            if not re.search(r"oof_preds\s*=\s*\(oof_prob[s]?\s*>?=", code):
                issues.append("Missing OOF thresholding: after selecting best_thr, compute oof_preds = (oof_prob(s) >= best_thr).astype(int) for METRICS")
        if re.search(r"(?m)^\s*oof_preds\[\s*val_idx\s*\]\s*=\s*.*predict_proba\(\s*X_va_sparse\s*\)", code) and not re.search(r"oof_preds\[\s*val_idx\s*\].*>=\s*0\.5", code):
            issues.append("OOF format bug: oof_preds[val_idx] must be binary labels (thresholded), not raw probabilities")

        # Using validation ensemble for test aggregation (often named ensemble_test even when it's val).
        if re.search(r"test_probs\s*\+\=\s*ensemble_test\s*/\s*N_SPLITS", code) and not re.search(r"predict_proba\(\s*X_te", code):
            issues.append("Incorrect test accumulation: ensemble_test appears to be computed from validation; compute proba*_test on X_te(_sparse) and accumulate that")
        if re.search(r"\bproba\d+_test\s*=\s*.*predict_proba\(\s*X_va_sparse\s*\)", code):
            issues.append("Incorrect test prediction source: proba*_test must be computed on X_te_sparse (test), not X_va_sparse (validation)")

        if "CalibratedClassifierCV(" in code and "from sklearn.calibration import CalibratedClassifierCV" not in code:
            issues.append("Missing import: from sklearn.calibration import CalibratedClassifierCV")
        if "MaxAbsScaler" in code and "from sklearn.preprocessing import MaxAbsScaler" not in code:
            issues.append("Missing import: from sklearn.preprocessing import MaxAbsScaler")
        if "LinearSVC" in code and "from sklearn.svm import LinearSVC" not in code:
            issues.append("Missing import: from sklearn.svm import LinearSVC")
        if "MultinomialNB" in code and "from sklearn.naive_bayes import MultinomialNB" not in code:
            issues.append("Missing import: from sklearn.naive_bayes import MultinomialNB")
        if "predict_proba(X_va)" in code and "X_va = hstack(" not in code:
            issues.append("Validation features not vectorized: expected X_va = hstack(...).tocsr() before predict_proba(X_va)")
        if "test_probs +=" not in code:
            # Allow any whitespace between name/operator.
            if not re.search(r"test_probs\s*\+\=", code):
                issues.append("Missing fold-wise test probability accumulation: test_probs += ... / N_SPLITS")
        if "test_probs += ensemble / N_SPLITS" in code and "predict_proba(X_te)" not in code:
            issues.append("Incorrect test accumulation: using validation ensemble for test_probs; use ensemble_test computed from X_te")
        if "skf.split(X_tr, y)" in code:
            issues.append("Incorrect CV split source: use skf.split(X, y), not skf.split(X_tr, y)")
        if "fit_transform(train_df['text'])" in code or 'fit_transform(train_df["text"])' in code:
            issues.append("Data leakage/mismatch: do not fit vectors on full train_df before CV; fit on fold train text (X_tr) inside loop")
        if "fit_transform(train_df['keyword'])" in code or 'fit_transform(train_df["keyword"])' in code:
            issues.append("Data leakage/mismatch: keyword vectorizer must fit on fold train subset, not full train_df")
        if "fit_transform(train_df['location'])" in code or 'fit_transform(train_df["location"])' in code:
            issues.append("Data leakage/mismatch: location vectorizer must fit on fold train subset, not full train_df")
        if "predict_proba(X_test)" in code:
            issues.append("Incorrect test input: sparse models must use predict_proba(X_te) after fold-specific vectorization, not raw X_test")
        if "clf1.fit(" in code and "clf1 =" not in code:
            issues.append("Missing classifier initialization: clf1 is used before assignment")
        if "clf2.fit(" in code and "clf2 =" not in code:
            issues.append("Missing classifier initialization: clf2 is used before assignment")
        if "clf3.fit(" in code and "clf3 =" not in code:
            issues.append("Missing classifier initialization: clf3 is used before assignment")

    if arch in ("CNN", "LSTM"):
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
    """Apply small deterministic fixes before invoking LLM repair."""
    if arch not in ("BoW_advanced", "BoW_advanced_thr"):
        return code

    # Strip any leaked prompt headings (must be code-only scripts).
    code_lines = code.splitlines()
    code_lines = [
        line
        for line in code_lines
        if not re.match(
            r"^(MANDATORY_SCRIPT_REQUIREMENTS|MANDATORY SCRIPT REQUIREMENTS|DATA LOADING|CROSS-VALIDATION|SUBMISSION|METRICS)\s*:\s*$",
            line.strip(),
        )
    ]
    code = "\n".join(code_lines)

    code = code.replace("import scipy.sparse.hstack", "")
    code = code.replace("from sklearn.calibrated_classifier_cv import CalibratedClassifierCV", "")
    code = code.replace("from sklearn.calibrated import CalibratedClassifierCV", "")

    # If DRY_RUN slices y before y is defined, insert y definition early.
    if "y = y[:200]" in code and not re.search(r"(?m)^\s*y\s*=\s*train_df\[['\"]target['\"]\]\.values", code):
        lines = code.splitlines()
        out: list[str] = []
        inserted = False
        for line in lines:
            out.append(line)
            if (not inserted) and re.match(r"^\s*train_df\s*=\s*pd\.read_csv", line):
                # insert after dataframes are loaded (best-effort)
                inserted = True
                out.append("y      = train_df['target'].values  # autofix: define y before DRY_RUN slicing")
        code = "\n".join(out)

    # If code uses X_text but only defined X, alias it.
    if re.search(r"\bX_tr_text\s*=\s*X_text\[", code) and not re.search(r"(?m)^\s*X_text\s*=", code) and re.search(r"(?m)^\s*X\s*=\s*train_df\[['\"]text['\"]\]\.values", code):
        code = re.sub(r"(?m)^X\s*=\s*train_df\[['\"]text['\"]\]\.values.*$", lambda m: m.group(0) + "\nX_text = X", code, count=1)

    code = code.replace(
        "from sklearn.preprocessing import MaxAbsScaler, LinearSVC",
        "from sklearn.preprocessing import MaxAbsScaler\nfrom sklearn.svm import LinearSVC",
    )

    # If validation predictions are made on raw X_va, create a sparse validation matrix and rewrite calls.
    if "predict_proba(X_va)" in code and "word_va = word_vec.transform(X_va)" not in code:
        val_block = (
            "\n    word_va = word_vec.transform(X_va)\n"
            "    char_va = char_vec.transform(X_va)\n"
            "    kw_va   = kw_vec.transform(X_va)\n"
            "    loc_va  = loc_vec.transform(X_va)\n"
            "    X_va_sparse = hstack([word_va, char_va, kw_va, loc_va]).tocsr()\n"
        )
        code = code.replace("proba1 = clf1.predict_proba(X_va)", val_block + "\n    proba1 = clf1.predict_proba(X_va_sparse)")
        code = code.replace("predict_proba(X_va)", "predict_proba(X_va_sparse)")

    # If we see the very common bug "X_va = hstack([word_tr,...])", rename to X_tr_sparse to reduce damage.
    code = re.sub(r"\bX_va\s*=\s*hstack\(\[word_tr,", "X_tr_sparse = hstack([word_tr,", code)
    code = re.sub(r"\bclf(\d)\.fit\(\s*X_tr\s*,\s*y_tr\s*\)", r"clf\1.fit(X_tr_sparse, y_tr)", code)

    # If X_va_sparse is built from *_tr_sparse, rewrite it to use transform(X_va_*).
    code = re.sub(
        r"(?m)^(?P<indent>\s*)X_va_sparse\s*=\s*hstack\(\[\s*word_tr_sparse\s*,\s*char_tr_sparse\s*,\s*kw_tr_sparse\s*,\s*loc_tr_sparse\s*\]\)\.tocsr\(\)\s*$",
        (
            r"\g<indent>X_va_sparse = hstack(["
            r"word_vec.transform(X_va_text), "
            r"char_vec.transform(X_va_text), "
            r"kw_vec.transform(X_va_kw), "
            r"loc_vec.transform(X_va_loc)"
            r"]).tocsr()"
        ),
        code,
    )

    # If X_tr_sparse is referenced but not defined, and we have *_tr_sparse, define it.
    if re.search(r"\bfit\(\s*X_tr_sparse\b", code) and not re.search(r"(?m)^\s*X_tr_sparse\s*=", code):
        if "word_tr_sparse" in code and "char_tr_sparse" in code and "kw_tr_sparse" in code and "loc_tr_sparse" in code:
            code = re.sub(
                r"(?m)^(?P<indent>\s*)loc_tr_sparse\s*=.*$",
                lambda m: m.group(0)
                + "\n"
                + m.group("indent")
                + "X_tr_sparse = hstack([word_tr_sparse, char_tr_sparse, kw_tr_sparse, loc_tr_sparse]).tocsr()",
                code,
                count=1,
            )

    # If test_probs is updated from a validation ensemble (ensemble_val), inject test ensemble computation.
    if "test_probs" in code and re.search(r"test_probs\s*\+\=\s*ensemble_val\s*/\s*N_SPLITS", code):
        code = re.sub(
            r"(?m)^(?P<indent>\s*)test_probs\s*\+\=\s*ensemble_val\s*/\s*N_SPLITS\s*$",
            (
                r"\g<indent>proba1_test = clf1.predict_proba(X_te_sparse)[:, 1]\n"
                r"\g<indent>proba2_test = clf2.predict_proba(X_te_sparse)[:, 1]\n"
                r"\g<indent>proba3_test = clf3.predict_proba(X_te_sparse)[:, 1]\n\n"
                r"\g<indent>ranked_test = [rankdata(p) / len(p) for p in [proba1_test, proba2_test, proba3_test]]\n"
                r"\g<indent>ensemble_test = np.mean(ranked_test, axis=0)\n\n"
                r"\g<indent>test_probs += ensemble_test / N_SPLITS"
            ),
            code,
        )

    # If the script uses X_test_sparse but defines X_te_sparse, rewrite to avoid NameError.
    if "X_test_sparse" in code and "X_te_sparse" in code and "X_test_sparse =" not in code:
        code = code.replace("X_test_sparse", "X_te_sparse")

    # If X_test_sparse is used but neither X_test_sparse nor X_te_sparse are defined, try to define X_te_sparse and rewrite.
    if "X_test_sparse" in code and "X_test_sparse =" not in code:
        if "X_te_sparse" not in code and "X_test_text" in code and "X_test_kw" in code and "X_test_loc" in code:
            code = code.replace("X_test_sparse", "X_te_sparse")
            lines = code.splitlines()
            out: list[str] = []
            inserted = False
            in_fold = False
            for line in lines:
                stripped = line.lstrip()
                indent = line[: len(line) - len(stripped)]
                if re.match(r"for\s+fold\s*,\s*\(train_idx,\s*val_idx\)\s+in\s+enumerate\(", stripped):
                    in_fold = True
                if in_fold and (not inserted) and stripped.startswith("X_va_sparse") and ").tocsr()" in stripped:
                    out.append(line)
                    out.append(
                        indent
                        + "X_te_sparse = hstack([word_vec.transform(X_test_text), char_vec.transform(X_test_text), kw_vec.transform(X_test_kw), loc_vec.transform(X_test_loc)]).tocsr()"
                    )
                    inserted = True
                    continue
                out.append(line)
            code = "\n".join(out)

    # If test probas are (incorrectly) computed on X_va_sparse, switch to X_te_sparse when available.
    if "X_te_sparse" in code:
        code = re.sub(
            r"(proba\d+_test\s*=\s*[^\\n]*predict_proba\()\s*X_va_sparse\s*(\)\s*\[:,\s*1\])",
            r"\\1X_te_sparse\\2",
            code,
        )

    # If X_te_sparse is referenced but never defined, insert a definition inside the fold after X_va_sparse is built.
    if "X_te_sparse" in code and not re.search(r"(?m)^\s*X_te_sparse\s*=", code):
        if "X_test_text" in code and "X_test_kw" in code and "X_test_loc" in code:
            lines = code.splitlines()
            out: list[str] = []
            inserted = False
            in_fold = False
            for line in lines:
                stripped = line.lstrip()
                indent = line[: len(line) - len(stripped)]
                if re.match(r"for\s+fold\s*,\s*\(train_idx,\s*val_idx\)\s+in\s+enumerate\(", stripped):
                    in_fold = True
                out.append(line)
                if in_fold and (not inserted) and re.match(r"^\s*X_va_sparse\s*=", line) and "tocsr()" in line:
                    out.append(
                        indent
                        + "X_te_sparse = hstack([word_vec.transform(X_test_text), char_vec.transform(X_test_text), kw_vec.transform(X_test_kw), loc_vec.transform(X_test_loc)]).tocsr()"
                    )
                    inserted = True
            code = "\n".join(out)

    # If oof_preds is incorrectly assigned from ensemble_test, replace with ensemble (validation) and inject validation ensemble block.
    if "oof_preds[val_idx]" in code and "ensemble_test" in code and "predict_proba(X_va_sparse)" not in code:
        lines = code.splitlines()
        out: list[str] = []
        in_fold = False
        injected = False
        for line in lines:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            if re.match(r"for\s+fold\s*,\s*\(train_idx,\s*val_idx\)\s+in\s+enumerate\(", stripped):
                in_fold = True
            # Replace the broken oof assignment line.
            if in_fold and re.match(r"^\s*oof_preds\[\s*val_idx\s*\]\s*=", line) and "ensemble_test" in line:
                out.append(indent + "oof_preds[val_idx] = (ensemble >= 0.5).astype(int)")
                continue

            out.append(line)

            # Inject right after the last fit call in the fold.
            if in_fold and (not injected) and re.match(r"^\s*clf3\.fit\(", stripped):
                injected = True
                out.append("")
                out.append(indent + "# VALIDATION PROBABILITIES (rank-averaged)")
                out.append(indent + "proba1 = clf1.predict_proba(X_va_sparse)[:, 1]")
                out.append(indent + "proba2 = clf2.predict_proba(X_va_sparse)[:, 1]")
                out.append(indent + "proba3 = clf3.predict_proba(X_va_sparse)[:, 1]")
                out.append("")
                out.append(indent + "ranked = [rankdata(p) / len(p) for p in [proba1, proba2, proba3]]")
                out.append(indent + "ensemble = np.mean(ranked, axis=0)")

        code = "\n".join(out)

    # If oof_preds is assigned a mean probability (/3), replace with rank-averaged validation ensemble + threshold.
    if arch == "BoW_advanced":
        code = re.sub(
            r"(?ms)^(?P<indent>\s*)oof_preds\[\s*val_idx\s*\]\s*=\s*\(.*?predict_proba\(\s*X_va_sparse\s*\)\s*\[:,\s*1\].*?\)\s*/\s*3\s*$",
            (
                r"\\g<indent># VALIDATION PROBABILITIES (rank-averaged)\\n"
                r"\\g<indent>proba1 = clf1.predict_proba(X_va_sparse)[:, 1]\\n"
                r"\\g<indent>proba2 = clf2.predict_proba(X_va_sparse)[:, 1]\\n"
                r"\\g<indent>proba3 = clf3.predict_proba(X_va_sparse)[:, 1]\\n\\n"
                r"\\g<indent>ranked = [rankdata(p) / len(p) for p in [proba1, proba2, proba3]]\\n"
                r"\\g<indent>ensemble = np.mean(ranked, axis=0)\\n"
                r"\\g<indent>oof_preds[val_idx] = (ensemble >= 0.5).astype(int)"
            ),
            code,
        )

        # If the script computes an "ensemble_test" from validation probabilities and uses it for both OOF and test,
        # split it into a validation ensemble (ensemble) and a true test ensemble (ensemble_test) from X_te_sparse.
        code = re.sub(
            r"(?ms)^(?P<indent>\s*)proba1\s*=\s*clf1\.predict_proba\(\s*X_va_sparse\s*\)\s*\[:,\s*1\]\s*\n"
            r"(?P=indent)proba2\s*=\s*clf2\.predict_proba\(\s*X_va_sparse\s*\)\s*\[:,\s*1\]\s*\n"
            r"(?P=indent)proba3\s*=\s*clf3\.predict_proba\(\s*X_va_sparse\s*\)\s*\[:,\s*1\]\s*\n"
            r"(?P=indent)#\s*Rank averaging\s*\n"
            r"(?P=indent)ranked\s*=\s*\[rankdata\(p\)\s*/\s*len\(p\)\s*for\s*p\s*in\s*\[proba1,\s*proba2,\s*proba3\]\]\s*\n"
            r"(?P=indent)ensemble_test\s*=\s*np\.mean\(ranked,\s*axis=0\)\s*\n"
            r"(?P=indent)#\s*Store oof_preds and update test_probs\s*\n"
            r"(?P=indent)oof_preds\[\s*val_idx\s*\]\s*=\s*\(ensemble_test\s*>=\s*0\.5\)\.astype\(int\)\s*\n"
            r"(?P=indent)test_probs\s*\+=\s*ensemble_test\s*/\s*N_SPLITS\s*$",
            (
                r"\g<indent># VALIDATION PROBABILITIES (rank-averaged)\n"
                r"\g<indent>proba1 = clf1.predict_proba(X_va_sparse)[:, 1]\n"
                r"\g<indent>proba2 = clf2.predict_proba(X_va_sparse)[:, 1]\n"
                r"\g<indent>proba3 = clf3.predict_proba(X_va_sparse)[:, 1]\n\n"
                r"\g<indent>ranked = [rankdata(p) / len(p) for p in [proba1, proba2, proba3]]\n"
                r"\g<indent>ensemble = np.mean(ranked, axis=0)\n\n"
                r"\g<indent>oof_preds[val_idx] = (ensemble >= 0.5).astype(int)\n\n"
                r"\g<indent># TEST PROBABILITIES (rank-averaged)\n"
                r"\g<indent>proba1_test = clf1.predict_proba(X_te_sparse)[:, 1]\n"
                r"\g<indent>proba2_test = clf2.predict_proba(X_te_sparse)[:, 1]\n"
                r"\g<indent>proba3_test = clf3.predict_proba(X_te_sparse)[:, 1]\n\n"
                r"\g<indent>ranked_test = [rankdata(p) / len(p) for p in [proba1_test, proba2_test, proba3_test]]\n"
                r"\g<indent>ensemble_test = np.mean(ranked_test, axis=0)\n\n"
                r"\g<indent>test_probs += ensemble_test / N_SPLITS"
            ),
            code,
        )

    # Rewrite common (incorrect) direct-probability averaging into rank-averaged ensemble_test update.
    code = re.sub(
        r"test_probs\s*\+=\s*\(\s*clf1\.predict_proba\(([^)]+)\)\s*\[:,\s*1\]\s*\+\s*\\?\s*"
        r"clf2\.predict_proba\(\1\)\s*\[:,\s*1\]\s*\+\s*\\?\s*"
        r"clf3\.predict_proba\(\1\)\s*\[:,\s*1\]\s*\)\s*/\s*N_SPLITS",
        "proba1_test = clf1.predict_proba(\\1)[:, 1]\n    proba2_test = clf2.predict_proba(\\1)[:, 1]\n    proba3_test = clf3.predict_proba(\\1)[:, 1]\n\n    ranked_test = [rankdata(p) / len(p) for p in [proba1_test, proba2_test, proba3_test]]\n    ensemble_test = np.mean(ranked_test, axis=0)\n\n    test_probs += ensemble_test / N_SPLITS",
        code,
        flags=re.M,
    )

    # If test_probs is incorrectly updated from the validation ensemble variable name (ensemble),
    # replace with fold-wise test rank-averaging on X_te_sparse when available.
    if "X_te_sparse" in code and re.search(r"(?m)^\s*test_probs\s*\+\=\s*ensemble\s*/\s*N_SPLITS\s*$", code):
        code = re.sub(
            r"(?m)^(?P<indent>\s*)test_probs\s*\+\=\s*ensemble\s*/\s*N_SPLITS\s*$",
            (
                r"\g<indent>proba1_test = clf1.predict_proba(X_te_sparse)[:, 1]\n"
                r"\g<indent>proba2_test = clf2.predict_proba(X_te_sparse)[:, 1]\n"
                r"\g<indent>proba3_test = clf3.predict_proba(X_te_sparse)[:, 1]\n\n"
                r"\g<indent>ranked_test = [rankdata(p) / len(p) for p in [proba1_test, proba2_test, proba3_test]]\n"
                r"\g<indent>ensemble_test = np.mean(ranked_test, axis=0)\n\n"
                r"\g<indent>test_probs += ensemble_test / N_SPLITS"
            ),
            code,
        )

    # If X_test is used only for length but never defined, fall back to X_test_text.
    if "len(X_test)" in code and not re.search(r"(?m)^\s*X_test\s*=", code) and re.search(r"(?m)^\s*X_test_text\s*=", code):
        code = code.replace("len(X_test)", "len(X_test_text)")

    # If y is defined as a bare Series, convert to .values (common LLM slip).
    code = re.sub(r"(?m)^\s*y\s*=\s*train_df\[['\"]target['\"]\]\s*$", "y      = train_df['target'].values", code)

    # If clf2 is a LinearSVC pipeline, wrap it with CalibratedClassifierCV so predict_proba exists.
    fixed_lines: list[str] = []
    for line in code.splitlines():
        if re.search(r"^\s*clf2\s*=\s*Pipeline\(\[.*LinearSVC.*\]\)\s*$", line) and "CalibratedClassifierCV" not in code:
            indent = re.match(r"^(\s*)", line).group(1)
            line = line.replace("clf2 = Pipeline([", "clf2 = CalibratedClassifierCV(estimator=Pipeline([")
            if line.rstrip().endswith("])"):
                line = line.rstrip()[:-2] + "]), cv=2)"
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    code = "\n".join(fixed_lines)

    required_imports = [
        ("CalibratedClassifierCV", "from sklearn.calibration import CalibratedClassifierCV"),
        ("MaxAbsScaler", "from sklearn.preprocessing import MaxAbsScaler"),
        ("LinearSVC", "from sklearn.svm import LinearSVC"),
        ("MultinomialNB", "from sklearn.naive_bayes import MultinomialNB"),
        ("rankdata(", "from scipy.stats import rankdata"),
        ("hstack(", "from scipy.sparse import hstack"),
    ]

    insert_lines: list[str] = []
    for token, imp in required_imports:
        if token in code and imp not in code:
            insert_lines.append(imp)

    if not insert_lines:
        return code

    # Insert missing imports right after the existing import block.
    lines = code.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("import ") or s.startswith("from ") or s == "":
            insert_at = i + 1
        else:
            break

    fixed = lines[:insert_at] + insert_lines + lines[insert_at:]
    return "\n".join(fixed)


def get_not_yet_successful(memory: ExperimentMemory) -> list[str]:
    successful_arches = {e["architecture"] for e in memory.experiments if e.get("success")}
    return [a for a in ARCH_SEQUENCE if a not in successful_arches]


def get_not_yet_attempted(memory: ExperimentMemory) -> list[str]:
    tried_arches = set(memory.get_tried_architectures())
    return [a for a in ARCH_SEQUENCE if a not in tried_arches]


def request_repair(
    llm: OllamaClient,
    arch: str,
    name: str,
    failed_code: str,
    stderr_text: str,
    stdout_text: str,
    attempt: int,
) -> str:
    extra_hint = ""
    if arch == "BoW_advanced" and "Validation features not vectorized" in stderr_text:
        extra_hint = (
            "BoW_advanced fix hint:\n"
            "Before predict_proba(X_va), transform and stack validation features:\n"
            "word_va = word_vec.transform(X_va)\n"
            "char_va = char_vec.transform(X_va)\n"
            "kw_va = kw_vec.transform(X_va)\n"
            "loc_va = loc_vec.transform(X_va)\n"
            "X_va = hstack([word_va, char_va, kw_va, loc_va]).tocsr()\n\n"
        )
    if arch == "BoW_advanced" and "inconsistent numbers of samples" in stderr_text:
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "Do NOT precompute X_tr from full train_df before CV.\n"
            "Use skf.split(X, y), then inside each fold compute:\n"
            "X_tr_txt = X[train_idx], X_va_txt = X[val_idx]\n"
            "Fit all vectorizers on X_tr_txt only, transform X_va_txt and X_test, then stack with hstack.\n"
            "This keeps feature rows aligned with y[train_idx]/y[val_idx].\n\n"
        )
    if arch == "BoW_advanced" and "Incorrect test accumulation" in stderr_text:
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "Compute test predictions inside each fold:\n"
            "  proba*_test = clf*.predict_proba(X_te_sparse)[:,1]\n"
            "  ranked_test = [rankdata(p)/len(p) for p in [proba1_test, proba2_test, proba3_test]]\n"
            "  ensemble_test = np.mean(ranked_test, axis=0)\n"
            "  test_probs += ensemble_test / N_SPLITS\n"
            "Never update test_probs using the validation ensemble.\n\n"
        )
    if arch == "BoW_advanced" and ("X_test_sparse" in stderr_text or "name 'X_test_sparse' is not defined" in stderr_text):
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "You used X_test_sparse but never defined it. Inside each fold, build the test matrix from the fold-fit vectorizers:\n"
            "  X_te_sparse = hstack([word_vec.transform(X_test_text), char_vec.transform(X_test_text), kw_vec.transform(X_test_kw), loc_vec.transform(X_test_loc)]).tocsr()\n"
            "Then use predict_proba(X_te_sparse).\n\n"
        )
    if arch == "BoW_advanced" and ("Missing validation prediction" in stderr_text or "Validation missing" in stderr_text or "OOF bug" in stderr_text):
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "Inside each fold, you MUST compute validation probabilities on X_va_sparse and store OOF:\n"
            "  proba1 = clf1.predict_proba(X_va_sparse)[:,1]\n"
            "  proba2 = clf2.predict_proba(X_va_sparse)[:,1]\n"
            "  proba3 = clf3.predict_proba(X_va_sparse)[:,1]\n"
            "  ranked = [rankdata(p)/len(p) for p in [proba1, proba2, proba3]]\n"
            "  ensemble = np.mean(ranked, axis=0)\n"
            "  oof_preds[val_idx] = (ensemble >= 0.5).astype(int)\n"
            "Do NOT use test predictions/ensemble_test for oof_preds.\n\n"
        )
    if arch in ("BoW_advanced", "BoW_advanced_thr") and ("Missing threshold tuning" in stderr_text or "Missing OOF thresholding" in stderr_text or "Missing OOF prob storage" in stderr_text):
        extra_hint += (
            "BoW_advanced threshold tuning hint:\n"
            "Store OOF probabilities (not labels) inside each fold:\n"
            "  oof_prob[val_idx] = ensemble_val\n"
            "After CV, find best_thr maximizing F1 on (oof_prob >= thr), then:\n"
            "  oof_preds = (oof_prob >= best_thr).astype(int)\n"
            "Use best_thr to threshold test_probs for submission.\n\n"
        )
    if arch == "BoW_advanced" and ("has no attribute 'predict_proba'" in stderr_text or "no attribute 'predict_proba'" in stderr_text):
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "LinearSVC does not support predict_proba. Use CalibratedClassifierCV:\n"
            "  clf2 = CalibratedClassifierCV(estimator=Pipeline([('sc', MaxAbsScaler()), ('svc', LinearSVC(...))]), cv=2)\n"
            "Then call clf2.predict_proba(...).\n\n"
        )
    if arch == "BoW_advanced" and "Missing OOF update" in stderr_text:
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "You must compute validation probabilities and fill oof_preds inside each fold:\n"
            "  proba1 = clf1.predict_proba(X_va_sparse)[:,1]\n"
            "  proba2 = clf2.predict_proba(X_va_sparse)[:,1]\n"
            "  proba3 = clf3.predict_proba(X_va_sparse)[:,1]\n"
            "  ranked = [rankdata(p)/len(p) for p in [proba1, proba2, proba3]]\n"
            "  ensemble = np.mean(ranked, axis=0)\n"
            "  oof_preds[val_idx] = (ensemble >= 0.5).astype(int)\n\n"
        )

    repair_prompt = (
        f"Repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS} for architecture: {arch}.\n"
        f"Keep submission path exactly: submissions/{name}_submission.csv\n\n"
        "Execution failed. Fix the script and return a fully executable Python script only.\n"
        "Do not include requirement prose, bullet lists, or copied prompt text inside the code block.\n\n"
        f"STDERR:\n{stderr_text}\n\n"
        f"{extra_hint}"
        f"STDOUT (tail):\n{tail(stdout_text, 30)}\n\n"
        "FAILED CODE:\n"
        "```python\n"
        f"{failed_code}\n"
        "```\n"
    )
    _, fixed_code = llm.propose(REPAIR_SYSTEM, repair_prompt)
    return fixed_code


def main(model: str, max_iterations: int, persist: bool = True):
    print("\n" + "=" * 60)
    print("  FULLY AUTONOMOUS ML RESEARCH AGENT - Disaster Tweets")
    print("=" * 60)

    os.makedirs("submissions", exist_ok=True)
    data_context = build_data_context()
    memory = ExperimentMemory(persist=persist)
    llm = OllamaClient(model=model)

    print(f"\n[Agent] Starting loop. Max iterations: {max_iterations}")
    print(f"[Agent] Target F1: {TARGET_F1}  |  Plateau window: {PLATEAU_WINDOW}\n")

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'-'*60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'-'*60}")

        tried = memory.get_tried_architectures()
        # Follow the fixed architecture sequence, without repeating already-attempted architectures.
        not_yet = get_not_yet_attempted(memory)
        best = memory.best()
        best_f1 = best["metrics"].get("f1", 0) if best else 0

        print(f"[Agent] Tried: {tried or 'none'}  |  Still needed (sequence): {not_yet or 'all covered'}")
        print(f"[Agent] Best F1: {best_f1:.5f}" if best_f1 else "[Agent] No successful runs yet")

        next_arch = not_yet[0] if not_yet else "BoW_advanced_thr"
        name = (
            f"iter{iteration:02d}_bowadvanced2"
            if next_arch == "BoW_advanced_thr"
            else f"iter{iteration:02d}_{next_arch.lower().replace('_', '')}"
        )

        if next_arch in ARCH_TEMPLATES:
            # Template-based: avoid fragile full-script generation for BoW variants.
            print(f"\n[THINK] Asking LLM to write model function for: {next_arch}...")
            model_prompt = get_model_prompt(next_arch)
            response, model_code = llm.propose(
                "You are a Python coding assistant. Write ONLY the requested function, nothing else.",
                model_prompt,
            )
            model_code = model_code or ""

            def _accept_function_only(code_text: str, fn_name: str, arg_count: int | None) -> bool:
                """
                Only accept code that defines exactly one top-level function.
                Prevents full-script injection into templates.
                """
                try:
                    tree = ast.parse(code_text)
                except SyntaxError:
                    return False

                body = list(tree.body)
                # Allow (optional) module docstring + one function def.
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                    body = body[1:]

                if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
                    return False
                if body[0].name != fn_name:
                    return False
                if arg_count is not None and len(body[0].args.args) != arg_count:
                    return False
                return True

            if model_code:
                if next_arch in ("BoW_advanced", "BoW_advanced_thr"):
                    if not _accept_function_only(model_code, "get_ensemble_probs", 4):
                        model_code = ""
                elif next_arch == "BoW":
                    if not _accept_function_only(model_code, "build_model", None):
                        model_code = ""
                elif next_arch in ("CNN", "LSTM"):
                    if not _accept_function_only(model_code, "build_model", 1):
                        model_code = ""

            code = fill_template(ARCH_TEMPLATES[next_arch], model_code, name)
            user_prompt = model_prompt
        else:
            print(f"\n[THINK] Asking LLM to write: {next_arch}...")
            arch_instruction = get_arch_prompt(next_arch)
            user_prompt = (
                f"Write a complete Python script implementing this architecture: {next_arch}\n\n"
                f"ARCHITECTURE DETAILS: {arch_instruction}\n\n"
                f"{data_context}\n"
                f"Experiment name for submission file: {name}\n"
                f"Submission path: submissions/{name}_submission.csv\n\n"
                f"Past experiments (do not repeat failures):\n{memory.get_history_summary()}\n"
            )
            response, code = llm.propose(FULL_SYSTEM, user_prompt)

        if not code:
            print("[THINK] LLM returned no code. Skipping iteration.")
            memory.add(
                name=f"iter{iteration:02d}_no_code", architecture="unknown",
                prompt_sent=user_prompt, code="", stdout="",
                stderr="LLM returned no code block", metrics={},
                analysis="LLM failed to produce a code block.", success=False,
            )
            continue

        arch = next_arch if next_arch in ARCH_TEMPLATES else infer_architecture(response + next_arch)
        print(f"[THINK] Architecture: {arch}  |  Name: {name}")

        attempt = 0
        result = None
        run_code = code
        while attempt <= MAX_REPAIR_ATTEMPTS:
            run_code = apply_light_autofixes(run_code, arch)
            issues = preflight_issues(run_code, arch)
            if issues:
                if attempt == MAX_REPAIR_ATTEMPTS:
                    result = {
                        "success": False,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": "Preflight validation failed:\n- " + "\n- ".join(issues),
                        "metrics": {},
                    }
                    break
                print(f"[REPAIR] Preflight issue detected. Asking LLM to repair (attempt {attempt + 1})...")
                fixed = request_repair(
                    llm=llm,
                    arch=arch,
                    name=name,
                    failed_code=run_code,
                    stderr_text="Preflight validation failed:\n- " + "\n- ".join(issues),
                    stdout_text="",
                    attempt=attempt + 1,
                )
                if not fixed:
                    result = {
                        "success": False,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": "Preflight validation failed:\n- " + "\n- ".join(issues),
                        "metrics": {},
                    }
                    break
                run_code = fixed
                attempt += 1
                continue

            ok, syntax_err = syntax_check(run_code)
            if not ok:
                if attempt == MAX_REPAIR_ATTEMPTS:
                    result = {
                        "success": False,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": f"SyntaxError: {syntax_err}",
                        "metrics": {},
                    }
                    break
                print(f"[REPAIR] Syntax issue detected. Asking LLM to repair (attempt {attempt + 1})...")
                fixed = request_repair(
                    llm=llm,
                    arch=arch,
                    name=name,
                    failed_code=run_code,
                    stderr_text=f"SyntaxError: {syntax_err}",
                    stdout_text="",
                    attempt=attempt + 1,
                )
                if not fixed:
                    result = {
                        "success": False,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": f"SyntaxError: {syntax_err}",
                        "metrics": {},
                    }
                    break
                run_code = fixed
                attempt += 1
                continue

            print("\n[EXECUTE] Running experiment...")
            result = run_experiment(run_code, name)
            if result["success"]:
                break

            if attempt == MAX_REPAIR_ATTEMPTS:
                break

            print(f"[REPAIR] Run failed. Asking LLM to repair (attempt {attempt + 1})...")
            fixed = request_repair(
                llm=llm,
                arch=arch,
                name=name,
                failed_code=run_code,
                stderr_text=result["stderr"],
                stdout_text=result["stdout"],
                attempt=attempt + 1,
            )
            if not fixed:
                break
            run_code = fixed
            attempt += 1

        code = run_code

        if not result:
            result = {
                "success": False,
                "timed_out": False,
                "stdout": "",
                "stderr": "Execution failed and repair returned no code.",
                "metrics": {},
            }

        if not result["success"]:
            print("\n--- FULL CODE ---")
            print(code)
            print("\n--- STDERR ---")
            print(result.get("stderr", ""))
            print("\n--- STDOUT ---")
            print(result.get("stdout", ""))

        # ── REFLECT ──────────────────────────────────────────────
        print("\n[REFLECT] Asking LLM to analyze results...")
        status = "success" if result["success"] else ("timeout" if result["timed_out"] else "crash")
        analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            name=name,
            status=status,
            metrics=result["metrics"] or "none",
            stdout_tail=tail(result["stdout"], 30),
            stderr_tail=tail(result["stderr"], 20),
        )
        analysis = llm.analyze(analysis_prompt)

        # ── Update memory ─────────────────────────────────────────
        memory.add(
            name=name, architecture=arch,
            prompt_sent=user_prompt, code=code,
            stdout=result["stdout"], stderr=result["stderr"],
            metrics=result["metrics"], analysis=analysis,
            success=result["success"],
        )

        f1 = result["metrics"].get("f1", None)
        print(f"\n[Result] {name} | F1={f1:.5f}" if f1 else f"\n[Result] {name} | FAILED ({status})")
        print(f"[Analysis] {analysis[:200].strip()}")

        if f1 and f1 >= TARGET_F1:
            print(f"\n[Agent] TARGET F1 {TARGET_F1} REACHED! Stopping early.")
            break
        if memory.is_plateau(PLATEAU_WINDOW, MIN_IMPROVEMENT):
            print(f"\n[Agent] Plateau detected. Stopping.")
            break

    memory.print_leaderboard()
    best = memory.best()
    if best:
        print(f"\n[Agent] Best submission: submissions/{best['name']}_submission.csv")
    print("\n[Agent] Done. Full log saved to experiment_log.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Autonomous ML Research Agent")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:3b")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore and do not write experiment_log.json (start sequence from scratch).",
    )
    args = parser.parse_args()
    if args.fresh:
        print("\n[Agent] Running in --fresh mode (no experiment_log.json read/write).")
    main(model=args.model, max_iterations=args.max_iter, persist=not args.fresh)
