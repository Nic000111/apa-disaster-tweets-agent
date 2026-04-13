import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import AGENT_V2_EXPERIMENT_LOG_PATH, PROJECT_ROOT


def load_autonomous_log(path: Path) -> list[dict]:
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []
    except (OSError, json.JSONDecodeError):
        return []


def find_log_path() -> Path | None:
    candidates = [
        AGENT_V2_EXPERIMENT_LOG_PATH,
        PROJECT_ROOT / "experiment_log.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_rows(rows: list[dict], log_path: Path) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    normalized = []
    for r in rows:
        metrics = r.get("metrics") or {}
        stderr = (r.get("stderr") or "").strip()
        success = bool(r.get("success", False))

        normalized.append(
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "architecture": r.get("architecture"),
                "timestamp": r.get("timestamp"),
                "success": success,
                "score": metrics.get("f1"),
                "accuracy": metrics.get("accuracy"),
                "error": "" if success else stderr,
                "analysis": r.get("llm_analysis") or "",
                "submission_file": f"submissions/{r.get('name')}_submission.csv" if r.get("name") else "",
                "log_file": str(log_path),
            }
        )

    df = pd.DataFrame(normalized)

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    return df


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "id",
        "timestamp",
        "name",
        "architecture",
        "success",
        "score",
        "accuracy",
        "submission_file",
        "error",
    ]
    existing = [c for c in columns if c in df.columns]
    out = df[existing].copy()
    if "score" in out.columns:
        out = out.sort_values("score", ascending=False, na_position="last")
    return out


st.set_page_config(page_title="Autonomous Agent Dashboard", layout="wide")
st.title("Disaster Tweets Autonomous Agent Dashboard")

log_path = find_log_path()
if not log_path:
    st.warning("No autonomous experiment log found. Expected experiment_log.json.")
    st.stop()

rows = load_autonomous_log(log_path)
all_df = normalize_rows(rows, log_path)

st.caption(f"Loaded log: {log_path}")

if all_df.empty:
    st.warning("Log exists but has no experiment rows.")
    st.stop()

architectures = sorted(all_df["architecture"].dropna().unique().tolist())
selected_arch = st.sidebar.multiselect(
    "Architectures",
    options=architectures,
    default=architectures,
)
show_failed = st.sidebar.checkbox("Show failed runs", value=True)

filtered_df = all_df.copy()
if selected_arch:
    filtered_df = filtered_df[filtered_df["architecture"].isin(selected_arch)]
if not show_failed:
    filtered_df = filtered_df[filtered_df["success"]]

successful_df = filtered_df[filtered_df["score"].notna()].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total runs", int(len(filtered_df)))
col2.metric("Successful runs", int(filtered_df["success"].sum()))
col3.metric("Failed runs", int((~filtered_df["success"]).sum()))
best_score = successful_df["score"].max() if not successful_df.empty else None
col4.metric("Best F1", f"{best_score:.4f}" if best_score is not None else "N/A")

st.subheader("Best Run")
if not successful_df.empty:
    best_row = successful_df.sort_values("score", ascending=False).iloc[0]
    st.json(best_row.dropna().to_dict())
else:
    st.info("No successful runs with metrics yet.")

st.subheader("F1 by Experiment")
if not successful_df.empty:
    chart_df = successful_df.copy()
    chart_df["label"] = (
        chart_df["id"].astype(str)
        + " | "
        + chart_df["architecture"].fillna("unknown").astype(str)
        + " | "
        + chart_df["name"].fillna("unnamed").astype(str)
    )
    fig = px.bar(
        chart_df.sort_values("score", ascending=False),
        x="label",
        y="score",
        color="architecture",
        hover_data=["name", "architecture", "score", "accuracy", "timestamp"],
        title="Autonomous experiment F1",
    )
    fig.update_layout(xaxis_title="Run", yaxis_title="F1")
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No F1 scores available to chart.")

st.subheader("Architecture Summary")
if not successful_df.empty:
    summary = (
        successful_df.groupby("architecture")["score"]
        .agg(["count", "max", "mean"])
        .reset_index()
        .rename(columns={"count": "runs", "max": "best_f1", "mean": "avg_f1"})
        .sort_values("best_f1", ascending=False)
    )
    st.dataframe(summary, width="stretch")
else:
    st.info("No successful runs for architecture summary.")

st.subheader("All Runs")
st.dataframe(display_table(filtered_df), width="stretch")

st.subheader("Failure Details")
failed_df = filtered_df[~filtered_df["success"]].copy()
if failed_df.empty:
    st.success("No failed runs in current filters.")
else:
    failed_cols = [c for c in ["id", "name", "architecture", "error"] if c in failed_df.columns]
    st.dataframe(failed_df[failed_cols], width="stretch")
