
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

def build_model():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    )



n_splits = 2 if DRY_RUN else 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(y))
test_preds = np.zeros(len(X_test_text))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y)):
    pipe = build_model()
    pipe.fit(X_text[tr_idx], y[tr_idx])
    oof_preds[va_idx]  = pipe.predict(X_text[va_idx])
    test_preds        += pipe.predict_proba(X_test_text)[:, 1] / n_splits

f1  = f1_score(y, oof_preds)
acc = accuracy_score(y, oof_preds)
print("OOF F1:", round(f1, 4), " Accuracy:", round(acc, 4))

if not DRY_RUN:
    sub = pd.DataFrame({"id": test_df["id"], "target": (test_preds >= 0.5).astype(int)})
    sub.to_csv("submissions/manual_bow_submission.csv", index=False)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')
