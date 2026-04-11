import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RANDOM_STATE,
    MAX_SIMPLE_EXPERIMENTS,
)

TRAIN_PATH = DATA_DIR / "train.csv"
LOG_PATH = LOGS_DIR / "experiments.jsonl"
BEST_RESULT_PATH = OUTPUTS_DIR / "best_result.json"

ALLOWED_MODEL_TYPES = ["avg_embed", "lstm"]
ALLOWED_VOCAB_SIZES = [5000, 10000]
ALLOWED_SEQ_LENGTHS = [50, 100]
ALLOWED_EMBED_DIMS = [32, 64]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH)
    df["text"] = df["text"].fillna("").astype(str)
    df["target"] = df["target"].astype(int)
    return df[["text", "target"]]


def load_experiment_history():
    if not LOG_PATH.exists():
        return []

    history = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def save_experiment_result(result: dict) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def ask_ollama_for_next_experiment(history: list[dict]) -> dict:
    if not history:
        history_text = "No previous experiments yet."
    else:
        history_text = json.dumps(history, indent=2)

    prompt = f"""
You are helping with a machine learning project on Kaggle Disaster Tweets.
We need the next experiment to try.

Allowed model_type values:
- avg_embed
- lstm

Allowed vocab_size values:
- 5000
- 10000

Allowed seq_length values:
- 50
- 100

Allowed embed_dim values:
- 32
- 64

Previous experiments:
{history_text}

Rules:
1. Pick ONE experiment only.
2. Do not repeat the exact same configuration if it already exists.
3. Prefer a reasonable next step.
4. Output ONLY valid JSON.
5. Use exactly this schema:

{{
  "model_type": "avg_embed",
  "vocab_size": 5000,
  "seq_length": 50,
  "embed_dim": 32,
  "reason": "short explanation"
}}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    output_text = response.json()["response"].strip()

    match = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in Ollama output:\n{output_text}")

    experiment = json.loads(match.group(0))
    return sanitize_experiment(experiment, history)


def sanitize_experiment(experiment: dict, history: list[dict]) -> dict:
    model_type = experiment.get("model_type", "avg_embed")
    if model_type not in ALLOWED_MODEL_TYPES:
        model_type = "avg_embed"

    vocab_size = experiment.get("vocab_size", 5000)
    if vocab_size not in ALLOWED_VOCAB_SIZES:
        vocab_size = 5000

    seq_length = experiment.get("seq_length", 50)
    if seq_length not in ALLOWED_SEQ_LENGTHS:
        seq_length = 50

    embed_dim = experiment.get("embed_dim", 32)
    if embed_dim not in ALLOWED_EMBED_DIMS:
        embed_dim = 32

    reason = str(experiment.get("reason", "No reason given."))

    clean = {
        "model_type": model_type,
        "vocab_size": vocab_size,
        "seq_length": seq_length,
        "embed_dim": embed_dim,
        "reason": reason
    }

    previous_configs = [
        {
            "model_type": h["model_type"],
            "vocab_size": h["vocab_size"],
            "seq_length": h["seq_length"],
            "embed_dim": h["embed_dim"],
        }
        for h in history
        if "model_type" in h
    ]

    current_config = {
        "model_type": clean["model_type"],
        "vocab_size": clean["vocab_size"],
        "seq_length": clean["seq_length"],
        "embed_dim": clean["embed_dim"],
    }

    if current_config in previous_configs:
        for mt in ALLOWED_MODEL_TYPES:
            for vs in ALLOWED_VOCAB_SIZES:
                for sl in ALLOWED_SEQ_LENGTHS:
                    for ed in ALLOWED_EMBED_DIMS:
                        candidate = {
                            "model_type": mt,
                            "vocab_size": vs,
                            "seq_length": sl,
                            "embed_dim": ed,
                            "reason": "Fallback config chosen to avoid repeat."
                        }
                        if {
                            "model_type": candidate["model_type"],
                            "vocab_size": candidate["vocab_size"],
                            "seq_length": candidate["seq_length"],
                            "embed_dim": candidate["embed_dim"],
                        } not in previous_configs:
                            return candidate

    return clean


def make_vectorizer_and_data(text_train, text_val, vocab_size, seq_length):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_length,
        standardize="lower_and_strip_punctuation",
    )

    train_ds_for_adapt = tf.data.Dataset.from_tensor_slices(text_train).batch(32)
    vectorizer.adapt(train_ds_for_adapt)

    x_train = vectorizer(np.array(text_train))
    x_val = vectorizer(np.array(text_val))

    return vectorizer, x_train, x_val


def build_model(model_type, vocab_size, seq_length, embed_dim):
    inputs = tf.keras.Input(shape=(seq_length,), dtype=tf.int64)

    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

    if model_type == "avg_embed":
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif model_type == "lstm":
        x = tf.keras.layers.LSTM(32)(x)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_experiment(experiment: dict, df: pd.DataFrame, experiment_number: int) -> dict:
    text_train, text_val, y_train, y_val = train_test_split(
        df["text"],
        df["target"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    _, x_train, x_val = make_vectorizer_and_data(
        text_train=text_train.tolist(),
        text_val=text_val.tolist(),
        vocab_size=experiment["vocab_size"],
        seq_length=experiment["seq_length"],
    )

    model = build_model(
        model_type=experiment["model_type"],
        vocab_size=experiment["vocab_size"],
        seq_length=experiment["seq_length"],
        embed_dim=experiment["embed_dim"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=1,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train,
        np.array(y_train),
        validation_data=(x_val, np.array(y_val)),
        epochs=3,
        batch_size=32,
        verbose=1,
        callbacks=callbacks,
    )

    val_probs = model.predict(x_val, verbose=0).flatten()
    val_preds = (val_probs >= 0.5).astype(int)
    val_f1 = f1_score(y_val, val_preds)

    result = {
        "experiment_number": experiment_number,
        "model_type": experiment["model_type"],
        "vocab_size": experiment["vocab_size"],
        "seq_length": experiment["seq_length"],
        "embed_dim": experiment["embed_dim"],
        "reason": experiment["reason"],
        "ollama_model": OLLAMA_MODEL,
        "val_f1": float(val_f1),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": float(history.history["val_accuracy"][-1]),
    }

    return result


def save_best_result(history: list[dict]) -> None:
    best = max(history, key=lambda x: x["val_f1"])
    with open(BEST_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)


def main():
    print("Loading data...")
    print(f"Using data dir: {DATA_DIR}")
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    df = load_data()
    print("Data loaded.")

    history = load_experiment_history()
    print(f"Loaded {len(history)} previous experiments.")

    for i in range(len(history) + 1, MAX_SIMPLE_EXPERIMENTS + 1):
        print(f"\n=== Experiment {i} ===")

        try:
            experiment = ask_ollama_for_next_experiment(history)
            print("Chosen experiment:")
            print(json.dumps(experiment, indent=2))

            result = run_experiment(experiment, df, i)
            print("Result:")
            print(json.dumps(result, indent=2))

            save_experiment_result(result)
            history.append(result)

        except Exception as e:
            error_result = {
                "experiment_number": i,
                "error": str(e),
                "ollama_model": OLLAMA_MODEL,
            }
            print("Experiment failed:")
            print(json.dumps(error_result, indent=2))
            save_experiment_result(error_result)
            history.append(error_result)

    valid_results = [h for h in history if "val_f1" in h]
    if valid_results:
        save_best_result(valid_results)
        print("\nBest result saved to outputs/best_result.json")
    else:
        print("\nNo valid completed experiments found.")


if __name__ == "__main__":
    main()
