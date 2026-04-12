"""Experiment-specific hooks for BoW_advanced architecture."""

import re

ARCH_PROMPT = """
Architecture: Multi-vectorizer TF-IDF ensemble with rank averaging.

CRITICAL RULES (follow exactly):
- N_SPLITS is already defined - do NOT redefine it or set it to any value.
- In the CV loop use y[train_index] and y[val_index], NEVER the full y array.
- Import scipy.stats at the top: from scipy.stats import rankdata
- oof_preds must be a 1D array: oof_preds = np.zeros(len(y))
- NEVER reference train_idx/val_idx outside the CV loop (no fitting/transforming before the loop).
- Define y = train_df['target'].values BEFORE any DRY_RUN slicing.

Step 1 - Feature matrix per fold (fit on train, transform train/val/test):
    from scipy.sparse import hstack
    word_vec = TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=2, max_df=0.97)
    char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6), sublinear_tf=True)
    kw_vec   = TfidfVectorizer(ngram_range=(1,2))
    loc_vec  = TfidfVectorizer(ngram_range=(1,2))
    IMPORTANT: build separate arrays BEFORE CV:
        X_text = train_df['text'].values
        X_kw   = train_df['keyword'].values
        X_loc  = train_df['location'].values
        X_test_text = test_df['text'].values
        X_test_kw   = test_df['keyword'].values
        X_test_loc  = test_df['location'].values
    In each fold, slice each field separately:
        X_tr_text = X_text[train_idx]; X_va_text = X_text[val_idx]
        X_tr_kw   = X_kw[train_idx];   X_va_kw   = X_kw[val_idx]
        X_tr_loc  = X_loc[train_idx];  X_va_loc  = X_loc[val_idx]
    Fit each vectorizer on the matching TRAIN field only:
        word/char: fit on X_tr_text
        kw: fit on X_tr_kw
        loc: fit on X_tr_loc
    Transform train/val/test separately and stack into NEW sparse matrices:
        X_tr_sparse, X_va_sparse, X_te_sparse (do not overwrite X_tr/X_va raw text variables).
    Stack: X_tr = hstack([word_tr, char_tr, kw_tr, loc_tr]).tocsr()

Step 2 - Train 3 classifiers on X_tr, y[train_index]:
    1. LogisticRegression(C=3.0, class_weight='balanced', solver='liblinear', max_iter=2000)
    2. CalibratedClassifierCV(estimator=Pipeline([('sc', MaxAbsScaler()), ('svc', LinearSVC(C=0.75, class_weight='balanced'))]), cv=2)
    3. MultinomialNB(alpha=0.05)

Step 3 - Rank averaging:
    from scipy.stats import rankdata
    ranked = [rankdata(p) / len(p) for p in [proba1, proba2, proba3]]
    ensemble = np.mean(ranked, axis=0)

Step 4 - Store: oof_preds[val_index] = (ensemble >= 0.5).astype(int)
Step 5 - Average test probs across folds, threshold at 0.5 for submission.
Important: inside each fold, compute test probabilities from clf1/clf2/clf3 on X_te,
rank-average those test probabilities, and add to test_probs using / N_SPLITS.

MUST: inside each fold, compute validation probabilities on X_va_sparse and set oof_preds[val_idx].
MUST: test_probs update uses ONLY the rank-averaged ensemble_test (do NOT sum model probabilities directly).
MUST: clf2 is CalibratedClassifierCV(Pipeline([('sc', MaxAbsScaler()), ('svc', LinearSVC(...))]), cv=2) so predict_proba works.

If DRY_RUN: limit train/val data to first 200 samples BEFORE the CV loop.
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

    loop_pos = code.find("for fold")
    if loop_pos != -1:
        pre_loop = code[:loop_pos]
        if re.search(r"\btrain_idx\b", pre_loop) or re.search(r"\bval_idx\b", pre_loop):
            issues.append("Index variables used before CV loop: never reference train_idx/val_idx outside the CV loop")

    if "y = y[:200]" in code and not re.search(r"(?m)^\s*y\s*=\s*train_df\[['\"]target['\"]\]\.values", code):
        issues.append("y sliced before definition: define y = train_df['target'].values before DRY_RUN slicing")

    if ("kw_vec" in code or "keyword" in code) and "train_df['keyword']" not in code and 'train_df["keyword"]' not in code:
        if "kw_vec" in code:
            issues.append("Keyword features missing: define X_kw = train_df['keyword'].values and fit kw_vec on X_tr_kw, not on X_tr text")
    if ("loc_vec" in code or "location" in code) and "train_df['location']" not in code and 'train_df["location"]' not in code:
        if "loc_vec" in code:
            issues.append("Location features missing: define X_loc = train_df['location'].values and fit loc_vec on X_tr_loc, not on X_tr text")

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

    if re.search(r"\bX_tr\s*,\s*X_va\s*=\s*X\[\s*train_idx\s*\]\s*,\s*X\[\s*val_idx\s*\]", code):
        if (
            re.search(r"\bclf1\.fit\(\s*X_tr\s*,\s*y_tr\s*\)", code)
            or re.search(r"\bclf2\.fit\(\s*X_tr\s*,\s*y_tr\s*\)", code)
            or re.search(r"\bclf3\.fit\(\s*X_tr\s*,\s*y_tr\s*\)", code)
        ):
            issues.append("Model fit bug: classifiers must be fit on the stacked sparse matrix (e.g., X_tr_sparse), not raw X_tr text")

    if code.count("test_probs +=") >= 2 and "ensemble_test" not in code:
        issues.append("Test aggregation bug: compute ensemble_test via rank averaging and do test_probs += ensemble_test / N_SPLITS (don't sum 3 model probs)")
    if "test_probs +=" in code and "ensemble_test" not in code and re.search(r"test_probs\s*\+=.*predict_proba\(", code):
        issues.append("Test aggregation bug: compute ensemble_test via rank averaging and do test_probs += ensemble_test / N_SPLITS (don't sum model probs directly)")

    if re.search(r"oof_preds\[\s*val_idx\s*\]\s*=\s*\(.*predict_proba\(", code, flags=re.S):
        if re.search(r"oof_preds\[\s*val_idx\s*\]\s*=\s*\(.*?\)\s*/\s*3", code, flags=re.S):
            issues.append("Validation aggregation bug: use rank averaging (rankdata + mean) for validation ensemble, not plain mean of probabilities")

    if "LinearSVC" in code and "predict_proba" in code and "CalibratedClassifierCV(" not in code:
        issues.append("Probability bug: LinearSVC has no predict_proba; wrap it with CalibratedClassifierCV(estimator=Pipeline([...LinearSVC...]), cv=2)")

    if re.search(r"(?m)^test_probs\s*\+=", code):
        issues.append("Incorrect test accumulation location: update test_probs inside the CV loop (indented), not after the loop")
    if ("X_te_sparse" in code or "X_te =" in code) and not re.search(r"predict_proba\(\s*X_te", code) and "ensemble_test" not in code:
        issues.append("Missing test prediction: compute proba*_test on X_te(_sparse), rank-average to ensemble_test, and accumulate into test_probs inside each fold")
    if re.search(r"test_probs\s*=\s*np\.zeros\(\s*len\(\s*X_test\s*\)\s*\)", code) and not re.search(r"(?m)^\s*X_test\s*=", code):
        issues.append("Undefined X_test: use X_test_text or define X_test = test_df['text'].values before using len(X_test)")
    if re.search(r"\bskf\.split\(\s*X_text\s*,\s*y\s*\)", code) and not re.search(r"(?m)^\s*y\s*=", code):
        issues.append("Missing target y: define y = train_df['target'].values before CV")

    if "X_test_sparse" in code and not re.search(r"(?m)^\s*X_test_sparse\s*=", code):
        issues.append("Undefined X_test_sparse: build X_te_sparse (or X_test_sparse) via vectorizers inside each fold before predict_proba")
    if re.search(r"predict_proba\(\s*X_te_sparse\s*\)", code) and not re.search(r"(?m)^\s*X_te_sparse\s*=", code):
        issues.append("Undefined X_te_sparse: build X_te_sparse inside each fold via hstack([...]).tocsr() before predict_proba(X_te_sparse)")

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
    if re.search(r"(?m)^\s*oof_preds\[\s*val_idx\s*\]\s*=\s*.*predict_proba\(\s*X_va_sparse\s*\)", code) and not re.search(r"oof_preds\[\s*val_idx\s*\].*>=\s*0\.5", code):
        issues.append("OOF format bug: oof_preds[val_idx] must be binary labels (thresholded), not raw probabilities")

    if re.search(r"test_probs\s*\+=\s*ensemble_test\s*/\s*N_SPLITS", code) and not re.search(r"predict_proba\(\s*X_te", code):
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
        if not re.search(r"test_probs\s*\+=", code):
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
