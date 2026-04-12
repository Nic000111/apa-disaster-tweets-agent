
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
    sub.to_csv("submissions/iter01_test_submission.csv", index=False)

print('METRICS: {"f1": ' + str(round(f1, 4)) + ', "accuracy": ' + str(round(acc, 4)) + '}')
