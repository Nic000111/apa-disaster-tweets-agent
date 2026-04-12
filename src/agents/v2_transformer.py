import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DATA_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    HF_DEFAULT_MODEL,
    RANDOM_STATE,
    MAX_V2_EXPERIMENTS,
    V2_TRANSFORMER_LOG_PATH,
    V2_TRANSFORMER_BEST_PATH,
)

TRAIN_PATH = DATA_DIR / "train.csv"
LOG_PATH = V2_TRANSFORMER_LOG_PATH
BEST_RESULT_PATH = V2_TRANSFORMER_BEST_PATH

SEED = RANDOM_STATE
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EXPERIMENTS = [
    {
        "name": "distilroberta_e3_lr2e5_wd001_bs16_len96",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 96,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "epochs": 3,
        "batch_size": 16,
    },
    {
        "name": "distilroberta_e4_lr2e5_wd001_bs16_len96",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 96,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "epochs": 4,
        "batch_size": 16,
    },
    {
        "name": "distilroberta_e3_lr3e5_wd001_bs16_len96",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 96,
        "lr": 3e-5,
        "weight_decay": 0.01,
        "epochs": 3,
        "batch_size": 16,
    },
    {
        "name": "distilroberta_e3_lr2e5_wd000_bs16_len128",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 128,
        "lr": 2e-5,
        "weight_decay": 0.00,
        "epochs": 3,
        "batch_size": 16,
    },
    {
        "name": "distilroberta_e4_lr2e5_wd000_bs16_len128",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 128,
        "lr": 2e-5,
        "weight_decay": 0.00,
        "epochs": 4,
        "batch_size": 16,
    },
    {
        "name": "distilroberta_e3_lr1e5_wd001_bs16_len128",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 128,
        "lr": 1e-5,
        "weight_decay": 0.01,
        "epochs": 3,
        "batch_size": 16,
    },
    {
        "name": "distilroberta_e3_lr2e5_wd001_bs8_len160",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 160,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "epochs": 3,
        "batch_size": 8,
    },
    {
        "name": "distilroberta_e4_lr3e5_wd001_bs8_len160",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 160,
        "lr": 3e-5,
        "weight_decay": 0.01,
        "epochs": 4,
        "batch_size": 8,
    },
    {
        "name": "distilroberta_e3_lr2e5_wd005_bs16_len128",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 128,
        "lr": 2e-5,
        "weight_decay": 0.05,
        "epochs": 3,
        "batch_size": 16,
    },
    {
        "name": "distilroberta_e5_lr2e5_wd001_bs16_len128",
        "model_name": HF_DEFAULT_MODEL,
        "max_length": 128,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "epochs": 5,
        "batch_size": 16,
    },
]

def save_jsonl(path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH)

    df["keyword"] = df["keyword"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["target"] = df["target"].astype(int)

    def combine(row):
        kw = row["keyword"].strip()
        txt = row["text"].strip()
        if kw:
            return f"[KEYWORD] {kw} [TEXT] {txt}"
        return txt

    df["model_text"] = df.apply(combine, axis=1)
    return df[["model_text", "target"]]


def make_hf_dataset(texts, labels):
    return Dataset.from_dict({
        "text": list(texts),
        "label": list(labels),
    })


def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
    return dataset.map(tokenize, batched=True)


def find_best_threshold(y_true, probs):
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))

    return best_threshold, float(best_f1)


def compute_metrics_eval(y_true, probs):
    threshold, best_f1 = find_best_threshold(y_true, probs)
    preds_05 = (probs >= 0.5).astype(int)
    f1_05 = f1_score(y_true, preds_05)
    return {
        "best_threshold": threshold,
        "val_f1_best_threshold": best_f1,
        "val_f1_at_0.5": float(f1_05),
    }


def train_one_experiment(exp: dict, df: pd.DataFrame, experiment_number: int) -> dict:
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["target"],
    )

    tokenizer = AutoTokenizer.from_pretrained(exp["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        exp["model_name"],
        num_labels=2
    )

    train_ds = make_hf_dataset(train_df["model_text"], train_df["target"])
    val_ds = make_hf_dataset(val_df["model_text"], val_df["target"])

    train_ds = tokenize_dataset(train_ds, tokenizer, exp["max_length"])
    val_ds = tokenize_dataset(val_ds, tokenizer, exp["max_length"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir = MODELS_DIR / exp["name"]

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=exp["lr"],
        per_device_train_batch_size=exp["batch_size"],
        per_device_eval_batch_size=exp["batch_size"],
        num_train_epochs=exp["epochs"],
        weight_decay=exp["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_total_limit=1,
        seed=SEED,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    predictions = trainer.predict(val_ds)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

    metrics = compute_metrics_eval(val_df["target"].values, probs)

    result = {
        "experiment_number": experiment_number,
        "name": exp["name"],
        "model_name": exp["model_name"],
        "max_length": exp["max_length"],
        "lr": exp["lr"],
        "weight_decay": exp["weight_decay"],
        "epochs": exp["epochs"],
        "batch_size": exp["batch_size"],
        "device": DEVICE,
        "val_f1_best_threshold": metrics["val_f1_best_threshold"],
        "best_threshold": metrics["best_threshold"],
        "val_f1_at_0.5": metrics["val_f1_at_0.5"],
    }

    return result


def main():
    print(f"Using device: {DEVICE}")
    print(f"Using data dir: {DATA_DIR}")
    print(f"Using HF model default: {HF_DEFAULT_MODEL}")
    df = load_data()
    print("Data loaded.")

    completed = []

    for i, exp in enumerate(EXPERIMENTS[:MAX_V2_EXPERIMENTS], start=1):
        print(f"\n=== Experiment {i}/{MAX_V2_EXPERIMENTS} ===")
        print(json.dumps(exp, indent=2))

        try:
            result = train_one_experiment(exp, df, i)
            print("RESULT:")
            print(json.dumps(result, indent=2))
            save_jsonl(LOG_PATH, result)
            completed.append(result)
        except Exception as e:
            error_row = {
                "experiment_number": i,
                "name": exp["name"],
                "error": str(e),
                "model_name": exp["model_name"],
            }
            print("FAILED:")
            print(json.dumps(error_row, indent=2))
            save_jsonl(LOG_PATH, error_row)

    valid = [x for x in completed if "val_f1_best_threshold" in x]
    if not valid:
        print("\nNo successful experiments.")
        return

    best = max(valid, key=lambda x: x["val_f1_best_threshold"])
    with open(BEST_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\nBest experiment:")
    print(json.dumps(best, indent=2))
    print(f"\nSaved best result to {BEST_RESULT_PATH}")


if __name__ == "__main__":
    main()
