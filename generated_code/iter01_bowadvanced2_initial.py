
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
warnings.filterwarnings("ignore")

DRY_RUN = os.environ.get("AGENT_DRY_RUN") == "1"
DATA_DIR = os.environ.get("DISASTER_AGENT_DATA_DIR", "data")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_df["keyword"]  = train_df["keyword"].fillna("")
train_df["location"] = train_df["location"].fillna("")
test_df["keyword"]   = test_df["keyword"].fillna("")
test_df["location"]  = test_df["location"].fillna("")

X_text      = train_df["text"].fillna("").astype(str).values
y           = train_df["target"].values
X_test_text = test_df["text"].fillna("").astype(str).values

if DRY_RUN:
    X_text = X_text[:200]
    y      = y[:200]

os.makedirs("submissions", exist_ok=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack
from scipy.stats import rankdata

def build_features(train_texts, valid_texts, test_texts, train_kw, valid_kw, test_kw,
                   train_loc, valid_loc, test_loc):
    word_vec = TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, min_df=2, max_df=0.97)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), sublinear_tf=True)
    kw_vec   = TfidfVectorizer(ngram_range=(1,2))
    loc_vec  = TfidfVectorizer(ngram_range=(1,2))
    tr = hstack([word_vec.fit_transform(train_texts), char_vec.fit_transform(train_texts),
                 kw_vec.fit_transform(train_kw), loc_vec.fit_transform(train_loc)]).tocsr()
    va = hstack([word_vec.transform(valid_texts), char_vec.transform(valid_texts),
                 kw_vec.transform(valid_kw), loc_vec.transform(valid_loc)]).tocsr()
    te = hstack([word_vec.transform(test_texts), char_vec.transform(test_texts),
                 kw_vec.transform(test_kw), loc_vec.transform(test_loc)]).tocsr()
    return tr, va, te

def get_ensemble_probs(X_tr, y_tr, X_va, X_te):
    models = [
        LogisticRegression(C=3.0, class_weight="balanced", solver="liblinear", max_iter=2000),
        CalibratedClassifierCV(
            estimator=make_pipeline(MaxAbsScaler(), LinearSVC(C=0.75, class_weight="balanced")),
            cv=2,
        ),
        MultinomialNB(alpha=0.05),
    ]
    vp, tp = [], []
    for m in models:
        m.fit(X_tr, y_tr)
        vp.append(m.predict_proba(X_va)[:, 1])
        tp.append(m.predict_proba(X_te)[:, 1])
    rank_avg = lambda ps: np.mean([rankdata(p) / len(p) for p in ps], axis=0)
    return {"valid": rank_avg(vp), "test": rank_avg(tp)}



kw_text      = train_df["keyword"].fillna("").astype(str).values
loc_text     = train_df["location"].fillna("").astype(str).values
kw_test_text = test_df["keyword"].fillna("").astype(str).values
loc_test_text = test_df["location"].fillna("").astype(str).values

if DRY_RUN:
    kw_text  = kw_text[:200]
    loc_text = loc_text[:200]

n_splits = 2 if DRY_RUN else 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_prob  = np.zeros(len(y))
test_probs = np.zeros(len(X_test_text))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y)):
    X_tr, X_va, X_te = build_features(
        X_text[tr_idx], X_text[va_idx], X_test_text,
        kw_text[tr_idx], kw_text[va_idx], kw_test_text,
        loc_text[tr_idx], loc_text[va_idx], loc_test_text,
    )
    fold_probs = get_ensemble_probs(X_tr, y[tr_idx], X_va, X_te)
    oof_prob[va_idx] = fold_probs["valid"]
    test_probs      += fold_probs["test"] / n_splits

# Tune threshold on OOF probabilities to maximize F1
thresholds = np.linspace(0.05, 0.95, 91)
best_thr = 0.5
best_f1 = -1.0
for thr in thresholds:
    preds = (oof_prob >= thr).astype(int)
    score = f1_score(y, preds)
    if score > best_f1:
        best_f1 = score
        best_thr = float(thr)

oof_preds = (oof_prob >= best_thr).astype(int)
f1  = f1_score(y, oof_preds)
acc = accuracy_score(y, oof_preds)
print("OOF best_thr:", round(best_thr, 3), " F1:", round(f1, 4), " Accuracy:", round(acc, 4))

if not DRY_RUN:
    sub = pd.DataFrame({"id": test_df["id"], "target": (test_probs >= best_thr).astype(int)})
    sub.to_csv("submissions/iter01_bowadvanced2_submission.csv", index=False)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')
