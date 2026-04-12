"""
templates.py â€” Architecture prompts and lightweight code templates.

Two modes:
- ARCH_PROMPTS: natural language descriptions used by agent_fully_autonomous.py
  (LLM generates full scripts guided by description + skeleton hints)
- ARCH_TEMPLATES: strict boilerplate used by agent_fully_autonomous.py for BoW variants
  (LLM only fills in the model function; everything else is guaranteed-correct)
"""

# â”€â”€ Shared constraints injected into every fully-autonomous prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SHARED_CONSTRAINTS = """
MANDATORY SCRIPT REQUIREMENTS:

IMPORTS â€” always include at the top of the script:
  import os, warnings, numpy as np, pandas as pd
  from sklearn.model_selection import StratifiedKFold
  from sklearn.metrics import f1_score, accuracy_score
  from sklearn.pipeline import Pipeline
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model import LogisticRegression
  warnings.filterwarnings('ignore')

RUNTIME FLAGS â€” always define once near the top:
    DRY_RUN = os.environ.get('AGENT_DRY_RUN') == '1'
    N_SPLITS = 2 if DRY_RUN else 5
    os.makedirs('submissions', exist_ok=True)
  Do NOT wrap these in a dict or other structure (no `RUNTIME_FLAGS = {...}`).

DATA LOADING â€” copy this pattern exactly:
    DATA_DIR = os.environ.get('DISASTER_AGENT_DATA_DIR', 'data')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv')).reset_index(drop=True)
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv')).reset_index(drop=True)
  train_df['keyword']  = train_df['keyword'].fillna('')
  train_df['location'] = train_df['location'].fillna('')
  test_df['keyword']   = test_df['keyword'].fillna('')
  test_df['location']  = test_df['location'].fillna('')
  X      = train_df['text'].values       # .values is MANDATORY â€” never use bare Series
  y      = train_df['target'].values     # .values is MANDATORY
  X_test = test_df['text'].values        # .values is MANDATORY
  If you need keyword/location features (e.g., BoW_advanced), ALSO define:
    X_kw      = train_df['keyword'].values
    X_loc     = train_df['location'].values
    X_test_kw  = test_df['keyword'].values
    X_test_loc = test_df['location'].values
  Then slice by fold indices (X_kw[train_idx], X_loc[val_idx], etc.).

DRY-RUN â€” add this block BEFORE the CV loop, AFTER loading data:
  if DRY_RUN:
      X = X[:200]
      y = y[:200]
      (and if you created X_kw/X_loc, slice them too)
    (Never use train_test_split for dry-run. Never set N_SPLITS=1.)
  NEVER special-case DRY_RUN inside the CV loop.

CROSS-VALIDATION â€” copy this skeleton:
  skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
  oof_preds  = np.zeros(len(y))         # 1D only
  test_probs = np.zeros(len(X_test))
  for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
      X_tr, X_va = X[train_idx], X[val_idx]
      y_tr, y_va = y[train_idx], y[val_idx]
      # ... fit model on X_tr, y_tr ...
      oof_preds[val_idx]  = model.predict(X_va)          # or predict_proba[:,1]
      test_probs         += model.predict_proba(X_test)[:,1] / N_SPLITS

METRICS â€” the VERY LAST printed line must be this exact format:
  f1  = f1_score(y, oof_preds)
  acc = accuracy_score(y, oof_preds)
  print('METRICS: {"f1": ' + str(round(f1,4)) + ', "accuracy": ' + str(round(acc,4)) + '}')
  (never use an f-string for this line â€” curly braces break f-strings)

SUBMISSION:
  sub = pd.DataFrame({'id': test_df['id'], 'target': (test_probs >= 0.5).astype(int)})
  sub.to_csv('submissions/__EXP_NAME___submission.csv', index=False)

FORBIDDEN â€” these will cause crashes, never write them:
  - import tensorflow / keras / tf.*
    - import scipy.sparse.hstack  (invalid import)
  - np.environ  (does not exist)
  - n_splits=1  (minimum is 2)
  - train_test_split instead of StratifiedKFold for dry-run
  - Bare pandas Series without .values (causes KeyError with StratifiedKFold)
  - CalibratedClassifierCV with cv > 2
  - oof_preds = np.zeros((len(y), N))  (must be 1D)
  - RUNTIME_FLAGS = {...}  (DRY_RUN/N_SPLITS must be plain variables)
"""

SHARED_REPAIR_CONSTRAINTS = """
REPAIR MODE REQUIREMENTS:
- Return a FULL corrected Python script inside one ```python code block.
- Keep the same target architecture family unless it is impossible to run.
- Fix only what is needed to make the script execute and print METRICS.
- Preserve submission output path: submissions/__EXP_NAME___submission.csv
- Never output explanations outside code.
- Never paste the prompt/requirements into the script (no headings like "DATA LOADING:" / "CROSS-VALIDATION:" / "SUBMISSION:" / "METRICS:" / "MANDATORY_*").
- If using sparse features, use scipy.sparse.hstack(...).tocsr() and NEVER np.hstack on sparse matrices.
- For sparse stacking, import exactly: from scipy.sparse import hstack
- Never use: import scipy.sparse.hstack
- Ensure predict_proba for test data uses the same feature pipeline as train/validation folds.
- Ensure test_probs is updated inside each CV fold (do not leave it all zeros).
- For BoW_advanced: do NOT fit models on raw text arrays. Build `X_tr_sparse`, `X_va_sparse`, `X_te_sparse` and fit/predict on those.
- For BoW_advanced: do NOT overwrite `X_tr`/`X_va` (raw text) with sparse matrices; use new variable names (`*_sparse`).
- For BoW_advanced: `kw_vec` must fit/transform keyword text and `loc_vec` must fit/transform location text (not the main tweet text).
- For BoW_advanced: update `test_probs` with the fold rank-averaged `ensemble_test` only (don’t sum 3 model probs directly).
- For BoW_advanced: if using LinearSVC, wrap it in CalibratedClassifierCV(cv=2) so `predict_proba` exists.
"""

# â”€â”€ Natural language architecture prompts (used by agent_fully_autonomous.py) â”€â”€
ARCH_PROMPTS = {
    "BoW_advanced": """
Architecture: Multi-vectorizer TF-IDF ensemble with rank averaging.

CRITICAL RULES (follow exactly):
- N_SPLITS is already defined â€” do NOT redefine it or set it to any value.
- In the CV loop use y[train_index] and y[val_index], NEVER the full y array.
- Import scipy.stats at the top: from scipy.stats import rankdata
- oof_preds must be a 1D array: oof_preds = np.zeros(len(y))
- NEVER reference train_idx/val_idx outside the CV loop (no fitting/transforming before the loop).
- Define y = train_df['target'].values BEFORE any DRY_RUN slicing.

Step 1 â€” Feature matrix per fold (fit on train, transform train/val/test):
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

Step 2 â€” Train 3 classifiers on X_tr, y[train_index]:
  1. LogisticRegression(C=3.0, class_weight='balanced', solver='liblinear', max_iter=2000)
    2. CalibratedClassifierCV(estimator=Pipeline([('sc', MaxAbsScaler()), ('svc', LinearSVC(C=0.75, class_weight='balanced'))]), cv=2)
  3. MultinomialNB(alpha=0.05)

Step 3 â€” Rank averaging:
  from scipy.stats import rankdata
  ranked = [rankdata(p) / len(p) for p in [proba1, proba2, proba3]]
  ensemble = np.mean(ranked, axis=0)

Step 4 â€” Store: oof_preds[val_index] = (ensemble >= 0.5).astype(int)
Step 5 â€” Average test probs across folds, threshold at 0.5 for submission.
Important: inside each fold, compute test probabilities from clf1/clf2/clf3 on X_te,
rank-average those test probabilities, and add to test_probs using / N_SPLITS.

MUST: inside each fold, compute validation probabilities on X_va_sparse and set oof_preds[val_idx].
MUST: test_probs update uses ONLY the rank-averaged `ensemble_test` (do NOT sum model probabilities directly).
MUST: clf2 is CalibratedClassifierCV(Pipeline([('sc', MaxAbsScaler()), ('svc', LinearSVC(...))]), cv=2) so predict_proba works.

If DRY_RUN: limit train/val data to first 200 samples BEFORE the CV loop.
""",

    "BoW_advanced_thr": """
Architecture: BoW_advanced + OOF threshold tuning.

Same as BoW_advanced (multi-vectorizer TF-IDF ensemble with rank averaging), PLUS:
- NEVER reference train_idx/val_idx outside the CV loop.
- Define y = train_df['target'].values BEFORE any DRY_RUN slicing.

THRESHOLD TUNING (required):
- Store OOF probabilities: `oof_prob[val_idx] = ensemble_val` (do not threshold inside the loop).
- After CV, search for `best_thr` in [0.05..0.95] maximizing F1 on `(oof_prob >= thr)`.
- Set `oof_preds = (oof_prob >= best_thr).astype(int)` and print METRICS as the last line.
- Use the same `best_thr` to threshold `test_probs` for the submission.
""",

    "BoW": """
Architecture: Simple TF-IDF + Logistic Regression pipeline.

Required imports (include ALL of these):
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model import LogisticRegression

Use sklearn Pipeline with:
- TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2, strip_accents='unicode')
- LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='liblinear')

Apply to the 'text' column only. Use predict() for OOF labels, predict_proba()[:,1] for test averaging.
""",

    "CNN": """
Architecture: 1D Convolutional Neural Network on token sequences using PyTorch.

Tokenization: fit CountVectorizer(max_features=10000) on training texts.
For each tweet, take the indices of present words (up to 100), pad to length 100 with zeros.

Model structure:
- nn.Embedding(10001, 64, padding_idx=0)
- Two parallel nn.Conv1d branches: kernel_size=3 and kernel_size=5, each with 128 filters
- Global max pooling on each branch (torch.max over sequence dim)
- Concatenate both branches â†’ 256 dims
- nn.Linear(256, 1) output (single logit, no sigmoid in forward)

Training: Adam(lr=1e-3), BCEWithLogitsLoss, batch_size=64, 3 epochs (1 if dry-run).
Use torch.cuda if available.
OOF: sigmoid(logits) >= 0.5 for binary preds. Test: average sigmoid probs across folds.
""",

    "LSTM": """
Architecture: Bidirectional LSTM on token sequences using PyTorch.

Tokenization: fit CountVectorizer(max_features=10000) on training texts.
For each tweet, take the indices of present words (up to 100), pad to length 100 with zeros.

Model structure:
- nn.Embedding(10001, 64, padding_idx=0)
- nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
- Take the last hidden state from both directions: torch.cat([h[-2], h[-1]], dim=1) â†’ 256 dims
- nn.Linear(256, 1) output (single logit, no sigmoid in forward)

Training: Adam(lr=1e-3), BCEWithLogitsLoss, batch_size=64, 3 epochs (1 if dry-run).
Use torch.cuda if available.
OOF: sigmoid(logits) >= 0.5 for binary preds. Test: average sigmoid probs across folds.
""",

    "Transformer": """
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
""",
}

# Keep a consistent architecture ordering when iterating / displaying prompts.
_ARCH_PROMPT_ORDER = ["BoW", "BoW_advanced", "BoW_advanced_thr", "CNN", "LSTM", "Transformer"]
ARCH_PROMPTS = {k: ARCH_PROMPTS[k] for k in _ARCH_PROMPT_ORDER if k in ARCH_PROMPTS}
# â”€â”€ Strict boilerplate templates (used by agent_fully_autonomous.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# These contain actual Python code with __MODEL_CODE__ and __EXP_NAME__ placeholders.
# The LLM only writes the model function; everything else is guaranteed correct.

SHARED_HEADER = """
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
"""

BOW_TEMPLATE = SHARED_HEADER + """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def build_model():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    )

__MODEL_CODE__

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
    sub.to_csv("submissions/__EXP_NAME___submission.csv", index=False)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')
"""

DEEP_TEMPLATE = SHARED_HEADER + """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer

vocab_size = 10000
max_len    = 100
tok = CountVectorizer(max_features=vocab_size, binary=False)
tok.fit(X_text)

def encode(texts):
    mat = tok.transform(texts).toarray()
    seqs = []
    for row in mat:
        idx = np.where(row > 0)[0][:max_len]
        pad = np.zeros(max_len, dtype=np.int64)
        pad[:len(idx)] = idx + 1
        seqs.append(pad)
    return np.array(seqs)

X_enc      = encode(X_text)
X_test_enc = encode(X_test_text)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def build_model(vocab_size):
    class DefaultModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(vocab_size, 64, padding_idx=0)
            self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
            self.fc   = nn.Linear(256, 1)
        def forward(self, x):
            _, (h, _) = self.lstm(self.emb(x))
            return self.fc(torch.cat([h[-2], h[-1]], dim=1))
    return DefaultModel()

__MODEL_CODE__

n_splits  = 2 if DRY_RUN else 5
n_epochs  = 1 if DRY_RUN else 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(y))
test_preds = np.zeros(len(X_test_enc))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_enc, y)):
    model     = build_model(vocab_size + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.BCEWithLogitsLoss()

    tr_loader = DataLoader(
        TensorDataset(torch.LongTensor(X_enc[tr_idx]), torch.FloatTensor(y[tr_idx])),
        batch_size=64, shuffle=True)
    va_loader = DataLoader(
        TensorDataset(torch.LongTensor(X_enc[va_idx]), torch.FloatTensor(y[va_idx])),
        batch_size=128)

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb).squeeze(), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        va_preds = torch.cat([
            torch.sigmoid(model(xb.to(device)).squeeze()).cpu()
            for xb, _ in va_loader
        ]).numpy()
        oof_preds[va_idx] = (va_preds >= 0.5).astype(int)

        te_loader = DataLoader(TensorDataset(torch.LongTensor(X_test_enc)), batch_size=128)
        te_preds  = torch.cat([
            torch.sigmoid(model(xb.to(device)).squeeze()).cpu()
            for (xb,) in te_loader
        ]).numpy()
        test_preds += te_preds / n_splits

f1  = f1_score(y, oof_preds)
acc = accuracy_score(y, oof_preds)
print("OOF F1:", round(f1, 4), " Accuracy:", round(acc, 4))

if not DRY_RUN:
    sub = pd.DataFrame({"id": test_df["id"], "target": (test_preds >= 0.5).astype(int)})
    sub.to_csv("submissions/__EXP_NAME___submission.csv", index=False)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')
"""

BOW_ADVANCED_TEMPLATE = SHARED_HEADER + """
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
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=2,
                               max_df=0.97, sublinear_tf=True, strip_accents="unicode")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=2,
                               sublinear_tf=True, strip_accents="unicode")
    kw_vec   = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1, sublinear_tf=True)
    loc_vec  = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1, sublinear_tf=True)
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
        CalibratedClassifierCV(estimator=make_pipeline(MaxAbsScaler(),
                               LinearSVC(C=0.75, class_weight="balanced")), cv=2),
        MultinomialNB(alpha=0.05),
    ]
    vp, tp = [], []
    for m in models:
        m.fit(X_tr, y_tr)
        vp.append(m.predict_proba(X_va)[:, 1])
        tp.append(m.predict_proba(X_te)[:, 1])
    rank_avg = lambda ps: np.mean([rankdata(p)/len(p) for p in ps], axis=0)
    return {"valid": rank_avg(vp), "test": rank_avg(tp)}

__MODEL_CODE__

kw_text      = train_df["keyword"].fillna("").astype(str).values
loc_text     = train_df["location"].fillna("").astype(str).values
kw_test_text = test_df["keyword"].fillna("").astype(str).values
lo_test_text = test_df["location"].fillna("").astype(str).values

if DRY_RUN:
    kw_text  = kw_text[:200]
    loc_text = loc_text[:200]

n_splits = 2 if DRY_RUN else 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_probs  = np.zeros(len(y))
test_probs = np.zeros(len(X_test_text))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y)):
    X_tr, X_va, X_te = build_features(
        X_text[tr_idx], X_text[va_idx], X_test_text,
        kw_text[tr_idx], kw_text[va_idx], kw_test_text,
        loc_text[tr_idx], loc_text[va_idx], lo_test_text,
    )
    fold_probs = get_ensemble_probs(X_tr, y[tr_idx], X_va, X_te)
    oof_probs[va_idx] = fold_probs["valid"]
    test_probs       += fold_probs["test"] / n_splits

oof_preds = (oof_probs >= 0.5).astype(int)
f1  = f1_score(y, oof_preds)
acc = accuracy_score(y, oof_preds)
print("OOF F1:", round(f1, 4), " Accuracy:", round(acc, 4))

if not DRY_RUN:
    sub = pd.DataFrame({"id": test_df["id"], "target": (test_probs >= 0.5).astype(int)})
    sub.to_csv("submissions/__EXP_NAME___submission.csv", index=False)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')
"""

BOW_ADVANCED_THR_TEMPLATE = SHARED_HEADER + """
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

__MODEL_CODE__

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
    sub.to_csv("submissions/__EXP_NAME___submission.csv", index=False)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')
"""

ARCH_TEMPLATES = {
    "BoW_advanced": BOW_ADVANCED_TEMPLATE,
    "BoW_advanced_thr": BOW_ADVANCED_THR_TEMPLATE,
    "BoW":  BOW_TEMPLATE,
    "CNN":  DEEP_TEMPLATE,
    "LSTM": DEEP_TEMPLATE,
}


def fill_template(template: str, model_code: str, name: str) -> str:
    """Replace __MODEL_CODE__ and __EXP_NAME__ placeholders."""
    return template.replace("__MODEL_CODE__", model_code).replace("__EXP_NAME__", name)


def get_model_prompt(arch: str) -> str:
    """Short function-only prompt used by agent_fully_autonomous.py (template mode)."""
    prompts = {
        "BoW_advanced": (
            "Write ONLY a Python function `get_ensemble_probs(X_tr, y_tr, X_va, X_te)` "
            "that trains LogisticRegression, CalibratedClassifierCV(LinearSVC), and MultinomialNB "
            "on sparse matrix X_tr, combines with rank averaging, and returns "
            "{'valid': probs, 'test': probs}. Write ONLY the function, no imports.\n\n"
            "```python\ndef get_ensemble_probs(X_tr, y_tr, X_va, X_te):\n    ...\n```"
        ),
        "BoW_advanced_thr": (
            "Write ONLY a Python function `get_ensemble_probs(X_tr, y_tr, X_va, X_te)` "
            "that trains LogisticRegression, CalibratedClassifierCV(LinearSVC), and MultinomialNB "
            "on sparse matrix X_tr, combines with rank averaging, and returns "
            "{'valid': probs, 'test': probs}. Write ONLY the function, no imports.\n\n"
            "```python\ndef get_ensemble_probs(X_tr, y_tr, X_va, X_te):\n    ...\n```"
        ),
        "BoW": (
            "Write ONLY a Python function `build_model()` returning a sklearn Pipeline with "
            "TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2) and "
            "LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000). "
            "Write ONLY the function, no imports.\n\n"
            "```python\ndef build_model():\n    return make_pipeline(...)\n```"
        ),
        "CNN": (
            "Write ONLY a Python function `build_model(vocab_size)` returning a PyTorch nn.Module "
            "with Embedding + two Conv1d branches (kernel 3 and 5) + global max pool + Linear output. "
            "Single logit output, no sigmoid. torch and nn already imported. "
            "Write ONLY the function.\n\n"
            "```python\ndef build_model(vocab_size):\n    class CNN(nn.Module): ...\n    return CNN()\n```"
        ),
        "LSTM": (
            "Write ONLY a Python function `build_model(vocab_size)` returning a PyTorch nn.Module "
            "with Embedding + BiLSTM(hidden=128, bidirectional=True) + last hidden state concat + Linear output. "
            "Single logit output, no sigmoid. torch and nn already imported. "
            "Write ONLY the function.\n\n"
            "```python\ndef build_model(vocab_size):\n    class BiLSTM(nn.Module): ...\n    return BiLSTM()\n```"
        ),
    }
    _MODEL_PROMPT_ORDER = ["BoW", "BoW_advanced", "BoW_advanced_thr", "CNN", "LSTM"]
    prompts = {k: prompts[k] for k in _MODEL_PROMPT_ORDER if k in prompts}
    return prompts.get(arch, "")
def get_arch_prompt(arch: str) -> str:
    """Full natural language architecture description used by agent_fully_autonomous.py."""
    return ARCH_PROMPTS.get(arch, "Propose an improved version of the best model so far.")
