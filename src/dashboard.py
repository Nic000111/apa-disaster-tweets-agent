import json

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    V1_SIMPLE_LOG_PATH,
    V2_TRANSFORMER_LOG_PATH,
    LEGACY_SIMPLE_LOG_PATH,
    LEGACY_ADVANCED_LOG_PATH,
)


def load_jsonl(path):
    if not path.exists():
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def prepare_simple_df(rows, *, agent_version: str, log_source: str):
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["agent_version"] = agent_version
    df["log_source"] = log_source

    if "val_f1" in df.columns:
        df["score"] = df["val_f1"]

    if "reason" not in df.columns:
        df["reason"] = ""

    if "name" not in df.columns:
        df["name"] = df.get("model_type", "simple_model")

    return df


def prepare_advanced_df(rows, *, agent_version: str, log_source: str):
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["agent_version"] = agent_version
    df["log_source"] = log_source

    if "val_f1_best_threshold" in df.columns:
        df["score"] = df["val_f1_best_threshold"]

    if "reason" not in df.columns:
        df["reason"] = ""

    return df


def safe_concat(dfs):
    valid = [df for df in dfs if not df.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True, sort=False)


def make_display_df(df):
    display_cols = [
        "agent_version",
        "log_source",
        "experiment_number",
        "name",
        "model_type",
        "model_name",
        "ollama_model",
        "vocab_size",
        "seq_length",
        "embed_dim",
        "max_length",
        "lr",
        "weight_decay",
        "epochs",
        "batch_size",
        "score",
        "val_f1",
        "val_f1_best_threshold",
        "val_f1_at_0.5",
        "best_threshold",
        "error",
        "reason",
    ]

    existing = [c for c in display_cols if c in df.columns]
    out = df[existing].copy()

    if "score" in out.columns:
        out = out.sort_values(by="score", ascending=False, na_position="last")

    return out


v1_df = prepare_simple_df(
    load_jsonl(V1_SIMPLE_LOG_PATH),
    agent_version="v1_simple",
    log_source="v1",
)
v1_legacy_df = prepare_simple_df(
    load_jsonl(LEGACY_SIMPLE_LOG_PATH),
    agent_version="v1_simple",
    log_source="legacy",
)

v2_df = prepare_advanced_df(
    load_jsonl(V2_TRANSFORMER_LOG_PATH),
    agent_version="v2_transformer",
    log_source="v2",
)
v2_legacy_df = prepare_advanced_df(
    load_jsonl(LEGACY_ADVANCED_LOG_PATH),
    agent_version="v2_transformer",
    log_source="legacy",
)

all_df = safe_concat([v1_df, v1_legacy_df, v2_df, v2_legacy_df])

st.set_page_config(page_title="Disaster Tweets Agent Dashboard", layout="wide")
st.title("Disaster Tweets Agent Dashboard")
st.write("This dashboard shows results from all agent versions.")

if all_df.empty:
    st.warning("No experiment logs found yet. Run an agent first.")
    st.stop()

agent_versions = sorted(all_df["agent_version"].dropna().unique().tolist())
selected_versions = st.sidebar.multiselect(
    "Select agent versions",
    options=agent_versions,
    default=agent_versions,
)

show_failed = st.sidebar.checkbox("Show failed experiments", value=True)

filtered_df = all_df[all_df["agent_version"].isin(selected_versions)].copy()

if not show_failed and "error" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["error"].isna()]

successful_df = filtered_df.copy()
if "score" in successful_df.columns:
    successful_df = successful_df[successful_df["score"].notna()]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total experiments", len(filtered_df))
col2.metric("Successful experiments", len(successful_df))

failed_count = 0
if "error" in filtered_df.columns:
    failed_count = filtered_df["error"].notna().sum()
col3.metric("Failed experiments", int(failed_count))

best_score = successful_df["score"].max() if len(successful_df) > 0 else None
col4.metric("Best score", f"{best_score:.4f}" if best_score is not None else "N/A")

st.subheader("Best Experiment")
if len(successful_df) > 0:
    best_row = successful_df.sort_values("score", ascending=False).iloc[0]
    st.json(best_row.dropna().to_dict())
else:
    st.info("No successful experiments yet.")

st.subheader("F1 Scores Across Experiments")
if len(successful_df) > 0:
    chart_df = successful_df.copy()

    if "name" not in chart_df.columns:
        chart_df["name"] = chart_df.get("model_type", "experiment")

    chart_df["label"] = (
        chart_df["agent_version"].astype(str)
        + " | exp "
        + chart_df["experiment_number"].astype(str)
        + " | "
        + chart_df["name"].fillna("unnamed").astype(str)
    )

    fig = px.bar(
        chart_df.sort_values("score", ascending=False),
        x="label",
        y="score",
        hover_data=[c for c in ["agent_version", "model_type", "model_name", "ollama_model", "score"] if c in chart_df.columns],
        title="Experiment scores",
    )
    fig.update_layout(xaxis_title="Experiment", yaxis_title="F1 score")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No score data available yet.")

st.subheader("Agent Version Comparison")
if len(successful_df) > 0:
    agent_summary = (
        successful_df.groupby("agent_version")["score"]
        .agg(["count", "max", "mean"])
        .reset_index()
        .rename(columns={"count": "num_runs", "max": "best_score", "mean": "avg_score"})
        .sort_values("best_score", ascending=False)
    )
    st.dataframe(agent_summary, use_container_width=True)
else:
    st.info("No successful experiments yet.")

st.subheader("All Experiments")
display_df = make_display_df(filtered_df)
st.dataframe(display_df, use_container_width=True)

if "error" in filtered_df.columns:
    failed_df = filtered_df[filtered_df["error"].notna()].copy()
    st.subheader("Failed Experiments")
    if failed_df.empty:
        st.success("No failed experiments in the selected view.")
    else:
        failed_cols = [c for c in ["agent_version", "experiment_number", "name", "error"] if c in failed_df.columns]
        st.dataframe(failed_df[failed_cols], use_container_width=True)
