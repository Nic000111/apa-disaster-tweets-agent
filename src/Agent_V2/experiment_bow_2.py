"""Experiment-specific hooks for BoW_advanced architecture."""

import re

ARCH_PROMPT = """
Architecture: BoW_advanced = multi-vectorizer TF-IDF ensemble with fold-wise rank averaging.

Implement a complete script. Code only.

Hard constraints:
- Do not redefine N_SPLITS.
- Define y = train_df['target'].values before any DRY_RUN slicing.
- If DRY_RUN is True, slice train-side arrays to first 200 samples BEFORE CV loop.
- Never use train_idx/val_idx outside CV loop.
- Use y_tr = y[train_idx], y_va = y[val_idx] inside loop.
- oof_preds must be 1D: np.zeros(len(y)).

Required imports:
- from scipy.sparse import hstack
- from scipy.stats import rankdata

Data arrays before CV:
- X_text = train_df['text'].values
- X_kw = train_df['keyword'].values
- X_loc = train_df['location'].values
- X_test_text = test_df['text'].values
- X_test_kw = test_df['keyword'].values
- X_test_loc = test_df['location'].values

Per-fold feature pipeline:
1. Slice by fold for each field (text/keyword/location).
2. Fit vectorizers on TRAIN subset only:
   - word_vec: TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=2, max_df=0.97)
   - char_vec: TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6), sublinear_tf=True)
   - kw_vec:   TfidfVectorizer(ngram_range=(1,2))
   - loc_vec:  TfidfVectorizer(ngram_range=(1,2))
3. Build sparse matrices:
   - X_tr_sparse = hstack([...]).tocsr()
   - X_va_sparse = hstack([...]).tocsr()
   - X_te_sparse = hstack([...]).tocsr()

Models (fit on X_tr_sparse, y_tr):
1. LogisticRegression(C=3.0, class_weight='balanced', solver='liblinear', max_iter=2000)
2. CalibratedClassifierCV(estimator=Pipeline([('sc', MaxAbsScaler()), ('svc', LinearSVC(C=0.75, class_weight='balanced'))]), cv=2)
3. MultinomialNB(alpha=0.05)

Ensembling and outputs:
- Validation: proba1/proba2/proba3 on X_va_sparse, rank-average, then set
  oof_preds[val_idx] = (ensemble_val >= 0.5).astype(int)
- Test: proba1_test/proba2_test/proba3_test on X_te_sparse, rank-average to ensemble_test,
  then test_probs += ensemble_test / N_SPLITS
- Do not directly sum raw test probabilities from models.

Final submission threshold is 0.5.
"""

MODEL_PROMPT = (
        "Return ONLY this function definition (no markdown, no imports, no extra text):\n"
        "def get_ensemble_probs(X_tr, y_tr, X_va, X_te): ...\n\n"
        "Requirements:\n"
        "- Inputs X_tr, X_va, X_te are already sparse matrices.\n"
        "- Fit exactly 3 models on X_tr, y_tr:\n"
        "  1) LogisticRegression(C=3.0, class_weight='balanced', solver='liblinear', max_iter=2000)\n"
        "  2) CalibratedClassifierCV(estimator=Pipeline([('sc', MaxAbsScaler()), ('svc', LinearSVC(C=0.75, class_weight='balanced'))]), cv=2)\n"
        "  3) MultinomialNB(alpha=0.05)\n"
        "- Compute valid probs on X_va and test probs on X_te for each model.\n"
        "- Use rank averaging with rankdata for both valid and test outputs.\n"
        "- Return dict with exact keys: {'valid': ensemble_valid, 'test': ensemble_test}.\n"
        "- Do not threshold; return probabilities in [0,1].\n"
)


def preflight_issues(code: str, arch: str) -> list[str]:
    issues: list[str] = []

    # Keep this intentionally lightweight: catch only high-impact, frequent failures.
    if "import scipy.sparse.hstack" in code:
        issues.append("Invalid import: use 'from scipy.sparse import hstack'")
    if "from sklearn.calibrated import CalibratedClassifierCV" in code or "from sklearn.calibrated_classifier_cv import CalibratedClassifierCV" in code:
        issues.append("Invalid import path for CalibratedClassifierCV; use 'from sklearn.calibration import CalibratedClassifierCV'")

    if re.search(r"(?m)^\s*y\s*=\s*train_df\[['\"]target['\"]\]\s*$", code):
        issues.append("Define y as numpy values: y = train_df['target'].values")
    if "y = y[:200]" in code and not re.search(r"(?m)^\s*y\s*=\s*train_df\[['\"]target['\"]\]\.values", code):
        issues.append("Define y before DRY_RUN slicing")
    if re.search(r"(?s)for\s+fold\s*,\s*\(train_idx,\s*val_idx\)\s+in\s+enumerate\([^\)]*\)\s*:\s*.*?\n\s*if\s+DRY_RUN\s*:", code):
        issues.append("Apply DRY_RUN slicing before the CV loop, not inside it")

    if "LinearSVC" in code and "predict_proba" in code and "CalibratedClassifierCV(" not in code:
        issues.append("LinearSVC must be wrapped in CalibratedClassifierCV for predict_proba")
    if "rankdata(" in code and "from scipy.stats import rankdata" not in code:
        issues.append("Missing import: from scipy.stats import rankdata")

    if re.search(r"\bproba\d+_test\s*=\s*.*predict_proba\(\s*X_va_sparse\s*\)", code):
        issues.append("Test probabilities must come from X_te_sparse, not X_va_sparse")
    if "test_probs" in code and not re.search(r"test_probs\s*\+=\s*ensemble_test\s*/\s*N_SPLITS", code):
        issues.append("Accumulate fold test predictions as: test_probs += ensemble_test / N_SPLITS")
    if "oof_preds" in code and not re.search(r"oof_preds\[\s*(?:val_idx|val_index)\s*\]\s*=", code):
        issues.append("Missing OOF assignment inside CV loop: oof_preds[val_idx] = ...")

    return issues


def apply_light_autofixes(code: str, arch: str) -> str:
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

    if "y = y[:200]" in code and not re.search(r"(?m)^\s*y\s*=\s*train_df\[['\"]target['\"]\]\.values", code):
        lines = code.splitlines()
        out: list[str] = []
        inserted = False
        for line in lines:
            out.append(line)
            if (not inserted) and re.match(r"^\s*train_df\s*=\s*pd\.read_csv", line):
                inserted = True
                out.append("y      = train_df['target'].values  # autofix: define y before DRY_RUN slicing")
        code = "\n".join(out)

    if re.search(r"\bX_tr_text\s*=\s*X_text\[", code) and not re.search(r"(?m)^\s*X_text\s*=", code) and re.search(r"(?m)^\s*X\s*=\s*train_df\[['\"]text['\"]\]\.values", code):
        code = re.sub(r"(?m)^X\s*=\s*train_df\[['\"]text['\"]\]\.values.*$", lambda m: m.group(0) + "\nX_text = X", code, count=1)

    code = code.replace(
        "from sklearn.preprocessing import MaxAbsScaler, LinearSVC",
        "from sklearn.preprocessing import MaxAbsScaler\nfrom sklearn.svm import LinearSVC",
    )

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

    code = re.sub(r"\bX_va\s*=\s*hstack\(\[word_tr,", "X_tr_sparse = hstack([word_tr,", code)
    code = re.sub(r"\bclf(\d)\.fit\(\s*X_tr\s*,\s*y_tr\s*\)", r"clf\1.fit(X_tr_sparse, y_tr)", code)

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

    if "X_test_sparse" in code and "X_te_sparse" in code and "X_test_sparse =" not in code:
        code = code.replace("X_test_sparse", "X_te_sparse")

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

    if "X_te_sparse" in code:
        code = re.sub(
            r"(proba\d+_test\s*=\s*[^\\n]*predict_proba\()\s*X_va_sparse\s*(\)\s*\[:,\s*1\])",
            r"\\1X_te_sparse\\2",
            code,
        )

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
            if in_fold and re.match(r"^\s*oof_preds\[\s*val_idx\s*\]\s*=", line) and "ensemble_test" in line:
                out.append(indent + "oof_preds[val_idx] = (ensemble >= 0.5).astype(int)")
                continue

            out.append(line)

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

    code = re.sub(
        r"test_probs\s*\+=\s*\(\s*clf1\.predict_proba\(([^)]+)\)\s*\[:,\s*1\]\s*\+\s*\\?\s*"
        r"clf2\.predict_proba\(\1\)\s*\[:,\s*1\]\s*\+\s*\\?\s*"
        r"clf3\.predict_proba\(\1\)\s*\[:,\s*1\]\s*\)\s*/\s*N_SPLITS",
        "proba1_test = clf1.predict_proba(\\1)[:, 1]\n    proba2_test = clf2.predict_proba(\\1)[:, 1]\n    proba3_test = clf3.predict_proba(\\1)[:, 1]\n\n    ranked_test = [rankdata(p) / len(p) for p in [proba1_test, proba2_test, proba3_test]]\n    ensemble_test = np.mean(ranked_test, axis=0)\n\n    test_probs += ensemble_test / N_SPLITS",
        code,
        flags=re.M,
    )

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

    if "len(X_test)" in code and not re.search(r"(?m)^\s*X_test\s*=", code) and re.search(r"(?m)^\s*X_test_text\s*=", code):
        code = code.replace("len(X_test)", "len(X_test_text)")

    code = re.sub(r"(?m)^\s*y\s*=\s*train_df\[['\"]target['\"]\]\s*$", "y      = train_df['target'].values", code)

    fixed_lines: list[str] = []
    for line in code.splitlines():
        if re.search(r"^\s*clf2\s*=\s*Pipeline\(\[.*LinearSVC.*\]\)\s*$", line) and "CalibratedClassifierCV" not in code:
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


def build_repair_hint(stderr_text: str) -> str:
    extra_hint = ""
    if "Validation features not vectorized" in stderr_text:
        extra_hint = (
            "BoW_advanced fix hint:\n"
            "Before predict_proba(X_va), transform and stack validation features:\n"
            "word_va = word_vec.transform(X_va)\n"
            "char_va = char_vec.transform(X_va)\n"
            "kw_va = kw_vec.transform(X_va)\n"
            "loc_va = loc_vec.transform(X_va)\n"
            "X_va = hstack([word_va, char_va, kw_va, loc_va]).tocsr()\n\n"
        )
    if "inconsistent numbers of samples" in stderr_text:
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "Do NOT precompute X_tr from full train_df before CV.\n"
            "Use skf.split(X, y), then inside each fold compute:\n"
            "X_tr_txt = X[train_idx], X_va_txt = X[val_idx]\n"
            "Fit all vectorizers on X_tr_txt only, transform X_va_txt and X_test, then stack with hstack.\n"
            "This keeps feature rows aligned with y[train_idx]/y[val_idx].\n\n"
        )
    if "Incorrect test accumulation" in stderr_text:
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "Compute test predictions inside each fold:\n"
            "  proba*_test = clf*.predict_proba(X_te_sparse)[:,1]\n"
            "  ranked_test = [rankdata(p)/len(p) for p in [proba1_test, proba2_test, proba3_test]]\n"
            "  ensemble_test = np.mean(ranked_test, axis=0)\n"
            "  test_probs += ensemble_test / N_SPLITS\n"
            "Never update test_probs using the validation ensemble.\n\n"
        )
    if "X_test_sparse" in stderr_text or "name 'X_test_sparse' is not defined" in stderr_text:
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "You used X_test_sparse but never defined it. Inside each fold, build the test matrix from the fold-fit vectorizers:\n"
            "  X_te_sparse = hstack([word_vec.transform(X_test_text), char_vec.transform(X_test_text), kw_vec.transform(X_test_kw), loc_vec.transform(X_test_loc)]).tocsr()\n"
            "Then use predict_proba(X_te_sparse).\n\n"
        )
    if "Missing validation prediction" in stderr_text or "Validation missing" in stderr_text or "OOF bug" in stderr_text:
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
    if "has no attribute 'predict_proba'" in stderr_text or "no attribute 'predict_proba'" in stderr_text:
        extra_hint += (
            "BoW_advanced fix hint:\n"
            "LinearSVC does not support predict_proba. Use CalibratedClassifierCV:\n"
            "  clf2 = CalibratedClassifierCV(estimator=Pipeline([('sc', MaxAbsScaler()), ('svc', LinearSVC(...))]), cv=2)\n"
            "Then call clf2.predict_proba(...).\n\n"
        )
    if "Missing OOF update" in stderr_text:
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
    return extra_hint


def get_arch_prompt() -> str:
    return ARCH_PROMPT


def get_model_prompt() -> str:
    return MODEL_PROMPT


def get_template(arch_templates: dict[str, str]) -> str | None:
    return arch_templates.get("BoW_advanced")
