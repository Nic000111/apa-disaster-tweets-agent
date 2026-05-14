"""Live Flask dashboard for Agent_4.

Three pages:
  /          F1 over time chart for past + current runs, error-rate stats
  /current   The currently running spec, the active prompt, the latest sweep decision
  /errors    Failed trials with outcome classification breakdown

No DB — everything is read from disk on each request (runs/, agent3_log.json).

Run:
  python3 src/Agent_4/dashboard.py
  # then open http://localhost:5000
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from typing import Any

from flask import Flask, jsonify, render_template_string, abort, send_from_directory

AGENT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(AGENT_ROOT, "docs")
LOGS_DIR = os.path.join(AGENT_ROOT, "data", "logs")
PROJECT_ROOT = os.path.abspath(os.path.join(AGENT_ROOT, "..", ".."))

# Nicc_2 layout: runs live under <repo>/runs/agent_4/<session>/. The "current"
# directory is the most recently invoked agent.py run. Fall back to the legacy
# src/Agent_4/runs/ path if a fresh in-place run wrote there.
_NICC2_RUNS = os.path.join(PROJECT_ROOT, "runs", "agent_4")
_LEGACY_RUNS = os.path.join(AGENT_ROOT, "runs")
RUNS_DIR = _NICC2_RUNS if os.path.isdir(_NICC2_RUNS) else _LEGACY_RUNS
# overall_best.json + sweep_decisions.jsonl live inside each session, so we
# look for them in the most recent "current" snapshot when present.
_CURRENT_RUN = os.path.join(_NICC2_RUNS, "current")
_OVERALL_CANDIDATES = [
    os.path.join(_CURRENT_RUN, "overall_best.json"),
    os.path.join(RUNS_DIR, "overall_best.json"),
]
_SWEEP_CANDIDATES = [
    os.path.join(_CURRENT_RUN, "sweep_decisions.jsonl"),
    os.path.join(RUNS_DIR, "sweep_decisions.jsonl"),
]
SWEEP_DECISIONS = next((p for p in _SWEEP_CANDIDATES if os.path.exists(p)), _SWEEP_CANDIDATES[-1])
OVERALL_BEST    = next((p for p in _OVERALL_CANDIDATES if os.path.exists(p)), _OVERALL_CANDIDATES[-1])

# The dashboard prefers the repo-local logs (committed snapshots under
# src/Agent_4/data/logs/) and falls back to absolute paths on Niccolò's / Simon's
# laptops if the repo-local copies aren't present.
def _first_existing(*candidates: str) -> str | None:
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


V1_LOG = os.environ.get("V1_LOG_PATH") or _first_existing(
    os.path.join(LOGS_DIR, "v1_experiment_log.json"),
    r"C:\Users\shoxx\Downloads\apa-disaster-tweets-agent\experiment_log.json",
)
V2_LOG_PATHS = [
    p for p in (
        os.environ.get("V2_LOG_PATHS").split(";")
        if os.environ.get("V2_LOG_PATHS")
        else [
            os.path.join(LOGS_DIR, "v2_agent_v2_log.json"),
            os.path.join(LOGS_DIR, "v2_git_clone_log.json"),
            r"C:\Users\shoxx\Downloads\Agent_V2\experiment_log.json",
            r"C:\Users\shoxx\Downloads\Git_Clone\apa-disaster-tweets-agent\experiment_log.json",
            r"C:\Users\shoxx\Downloads\Git_Clone\apa-disaster-tweets-agent-1\experiment_log.json",
        ]
    )
    if p and p.strip()
]
AGENT3_LOG = os.environ.get("V3_LOG_PATH") or _first_existing(
    os.path.join(LOGS_DIR, "v3_agent3_log.json"),
    os.path.join(PROJECT_ROOT, "agent3_log.json"),
)

app = Flask(__name__)


# ----------------------------- data loaders ----------------------------- #

def load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_text(path: str, max_bytes: int = 200_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = fh.read(max_bytes)
        return data
    except FileNotFoundError:
        return ""


def gather_trials() -> list[dict[str, Any]]:
    """Walk all Agent_4 session summaries + the historical Agent_3 log."""
    trials: list[dict[str, Any]] = []

    # Agent_4 runs → V4. Nicc_2 layout has two levels of sessions
    # (runs/agent_4/<bucket>/<session>/summary.json), so glob both depths.
    summary_glob = sorted(set(
        glob(os.path.join(RUNS_DIR, "*", "summary.json")) +
        glob(os.path.join(RUNS_DIR, "*", "*", "summary.json"))
    ))
    for summary_path in summary_glob:
        summary = load_json(summary_path) or {}
        session = os.path.basename(os.path.dirname(summary_path))
        for trial in summary.get("trials", []):
            run_idx = trial.get("run_index")
            run_dir_rel = f"{session}/run_{run_idx:03d}" if isinstance(run_idx, int) and run_idx > 0 else None
            trials.append({
                "version": "V4",
                "source": "agent4",
                "session": session,
                "phase": summary.get("phase", "sweep"),
                "family": summary.get("family", "unknown"),
                "family_key": summary.get("family_key", ""),
                "run_index": run_idx,
                "success": bool(trial.get("success", False)),
                "f1": (trial.get("metrics") or {}).get("f1"),
                "accuracy": (trial.get("metrics") or {}).get("accuracy"),
                "outcome": trial.get("outcome", "unknown"),
                "error_summary": trial.get("error_summary", ""),
                "wall_seconds": trial.get("wall_seconds"),
                "repair_attempts": trial.get("repair_attempts", 0),
                "run_dir": run_dir_rel,
                "timestamp": None,
            })

    # Historical Agent_3 log (single flat list)  → V3
    historical = load_json(AGENT3_LOG)
    if isinstance(historical, list):
        for entry in historical:
            metrics = entry.get("metrics") or {}
            trials.append({
                "version": "V3",
                "source": "agent3-history",
                "session": entry.get("run_name", "history"),
                "phase": "history",
                "family": entry.get("family", "unknown"),
                "family_key": "",
                "run_index": entry.get("run_index"),
                "success": bool(entry.get("success", False)),
                "f1": metrics.get("f1"),
                "accuracy": metrics.get("accuracy"),
                "outcome": "success" if entry.get("success") else "training_crash",
                "error_summary": "",
                "wall_seconds": None,
                "repair_attempts": 0,
                "run_dir": None,
                "timestamp": entry.get("timestamp"),
            })

    # V1 and V2 flat experiment_log.json files (same schema, different folders).
    # V2 may have multiple snapshots — dedupe by (name, timestamp).
    seen_v2: set[tuple[str, str]] = set()
    log_sources: list[tuple[str, str]] = [("V1", V1_LOG)]
    for p in V2_LOG_PATHS:
        log_sources.append(("V2", p))

    for version_label, log_path in log_sources:
        data = load_json(log_path)
        if not isinstance(data, list):
            continue
        for entry in data:
            metrics = entry.get("metrics") or {}
            stderr = entry.get("stderr") or ""
            code_gen = entry.get("code_generated") or ""
            success = bool(entry.get("success", False))
            if success:
                outcome = "success"
            elif "Preflight validation failed" in stderr:
                outcome = "preflight_failed"
            elif "TIMEOUT" in stderr:
                outcome = "timeout"
            elif "FileNotFoundError" in stderr:
                outcome = "file_not_found"
            elif "[LLM ERROR]" in code_gen or "LLM failed" in (entry.get("llm_analysis") or ""):
                outcome = "code_gen_failed"
            elif "MISSING_METRICS_LINE" in stderr:
                outcome = "no_metrics"
            elif "Traceback" in stderr:
                outcome = "training_crash"
            else:
                outcome = "unknown_failure"

            key = (entry.get("name", "?"), entry.get("timestamp", ""))
            if version_label == "V2":
                if key in seen_v2:
                    continue
                seen_v2.add(key)

            trials.append({
                "version": version_label,
                "source": version_label.lower(),
                "session": entry.get("name", "?"),
                "phase": "baseline",
                "family": entry.get("architecture", "unknown"),
                "family_key": "",
                "run_index": entry.get("id"),
                "success": success,
                "f1": metrics.get("f1"),
                "accuracy": metrics.get("accuracy"),
                "outcome": outcome,
                "error_summary": (stderr.splitlines() or [""])[-1][:200],
                "wall_seconds": None,
                "repair_attempts": 0,
                "run_dir": None,
                "timestamp": entry.get("timestamp"),
            })
    return trials


def latest_session_dir() -> str | None:
    sessions = [
        d for d in glob(os.path.join(RUNS_DIR, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "summary.json")) is False
        and not d.endswith(".jsonl")
    ]
    candidates = [
        d for d in glob(os.path.join(RUNS_DIR, "*"))
        if os.path.isdir(d)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def latest_run_dir(session_dir: str) -> str | None:
    if not session_dir:
        return None
    run_dirs = [d for d in glob(os.path.join(session_dir, "run_*")) if os.path.isdir(d)]
    if not run_dirs:
        return None
    return max(run_dirs, key=os.path.getmtime)


def tail_jsonl(path: str, n: int = 20) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-n:]


# ----------------------------- routes ----------------------------- #

DASHBOARD_HTML = r"""
<!doctype html>
<html><head>
<meta charset="utf-8"><title>Agent_4 Dashboard</title>
<meta http-equiv="refresh" content="10">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; background: #f7f8fa; color: #222; }
  header { background: #2b8a3e; color: white; padding: 14px 24px; }
  header h1 { margin: 0; font-size: 20px; }
  nav a { color: white; margin-right: 18px; text-decoration: none; font-weight: 600; }
  nav a:hover { text-decoration: underline; }
  main { padding: 20px 24px; max-width: 1200px; margin: 0 auto; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
  .card { background: white; border-radius: 8px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); flex: 1; min-width: 160px; }
  .card h3 { margin: 0 0 6px; font-size: 12px; text-transform: uppercase; color: #666; letter-spacing: 0.05em; }
  .card .value { font-size: 28px; font-weight: 700; }
  .card.good .value { color: #2b8a3e; }
  .card.bad  .value { color: #c0392b; }
  .card.neutral .value { color: #2b5da4; }
  .chart-card { background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }
  th { background: #fafbfc; font-weight: 600; }
  .pill { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; display: inline-block; }
  .pill.ok { background: #e0f5e8; color: #2b8a3e; }
  .pill.err { background: #fde2e2; color: #a83232; }
  .pill.warn { background: #fff8d6; color: #8a6d00; }
  pre.code { background: #0d1117; color: #c9d1d9; padding: 14px; border-radius: 6px; overflow: auto; max-height: 480px; font-size: 12px; }
  .small { color: #777; font-size: 12px; }
</style></head>
<body>
<header>
  <h1>Agent_4 — Live Dashboard</h1>
  <nav>
    <a href="/">Overview</a>
    <a href="/current">Current Input</a>
    <a href="/errors">Errors</a>
    <a href="/architectures">Architectures</a>
    <a href="/comparison">Comparison</a>
  </nav>
</header>
<main>
{{ body|safe }}
</main>
</body></html>
"""


def render(body: str) -> str:
    return render_template_string(DASHBOARD_HTML, body=body)


@app.route("/")
def overview():
    trials = gather_trials()
    successful = [t for t in trials if t["success"] and isinstance(t["f1"], (int, float))]
    failed = [t for t in trials if not t["success"]]
    total = len(trials)
    error_rate = (len(failed) / total * 100) if total else 0.0

    # Best so far across all sources
    best = max(successful, key=lambda t: t["f1"]) if successful else None

    # Color = version, shape = data type:
    #   circle = validation F1 success     (each version's JSON log)
    #   diamond = Kaggle public score      (only if recorded in overall_best.json's kaggle_submission)
    #   red X  = failure (validation F1 NA)
    version_order = {"V1": 0, "V2": 1, "V3": 2, "V4": 3}
    all_sorted = sorted(
        trials,
        key=lambda t: (version_order.get(t.get("version", "Z"), 9), t.get("session", ""), t.get("run_index") or 0),
    )
    series_ok: dict[str, list[dict[str, Any]]] = defaultdict(list)
    series_fail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, t in enumerate(all_sorted, start=1):
        v = t.get("version", "?")
        if t.get("success") and isinstance(t.get("f1"), (int, float)):
            series_ok[v].append({
                "x": idx,
                "y": round(t["f1"], 4),
                "label": f"✓ {t.get('family','?')} · {t.get('session','?')}#{t.get('run_index','')}",
            })
        else:
            series_fail[v].append({
                "x": idx,
                "y": -0.05,  # below the F1 axis range so failures form their own row
                "label": f"✗ {t.get('outcome','failure')} · {t.get('family','?')} · {t.get('session','?')}",
            })

    # Kaggle public scores transcribed from Niccolò's Kaggle submissions page,
    # matched to the agent run that produced each submission CSV.
    # Mapping based on: submission date vs commit/run dates, filename family hints,
    # and Kaggle-vs-validation F1 proximity (transformers typically gain +0.01–0.03 on Kaggle).
    # Schema: (version, score, filename, days_ago, matched_run, comment)
    KAGGLE_SCORES = [
        # V3 BERTweet variants (16-17d ago, around the BERTweet commit 2026-04-27)
        ("V3", 0.84216, "bertweet_long_regularized_submission.csv",   16, "V3 BERTweet (long+regularized variant)", ""),
        ("V3", 0.83665, "bertweet_long_regularized_submission_2.csv", 16, "V3 BERTweet (long+regularized v2)",      ""),
        ("V3", 0.84002, "bertweet_best_overall_submission.csv",       17, "V3 BERTweet best",                        ""),
        ("V3", 0.84002, "best_overall_submission.csv (Simon)",        17, "V3 BERTweet (re-upload)",                 ""),
        ("V3", 0.78087, "best_overall_submission.csv",                17, "V3 sweep best (BERTweet baseline)",       "duplicate score"),
        ("V3", 0.78087, "best_overall_submission.csv",                17, "V3 sweep best (BERTweet baseline)",       "duplicate score"),
        ("V3", 0.78087, "best_overall_submission.csv",                14, "V3 'Agent_3 test submit'",                ""),
        # V3 with RoBERTa (22d ago — RoBERTa commit was 2026-04-21)
        ("V3", 0.83420, "best_overall_submission.csv",                22, "V3 RoBERTa best",                          ""),
        ("V3", 0.82776, "best_overall_submission.csv (Simon)",        16, "V3 (RoBERTa-era)",                         ""),
        # V3 early refinement (23d ago — first big commit)
        ("V3", 0.82837, "submission.csv",                             23, "V3 early refinement (RoBERTa)",            ""),
        ("V3", 0.80324, "submission.csv",                             23, "V3 early refinement",                      ""),
        ("V3", 0.80324, "submission.csv",                             23, "V3 early refinement (re-upload)",          ""),
        # V3 May 7 sweep — direct match to the trials in agent3_log.json
        ("V3", 0.79466, "direct_final_submission_test.csv",            7, "V3 BERTweet sweep · val F1 0.7834 (matches bertweet_20260507_230835_run_01)", ""),
        # 1d-ago uploads — file naming submission_1/2/3.csv suggests RoBERTa ensemble variants;
        # F1 0.82+ matches V3 RoBERTa val F1 (0.8224 / 0.8196 / 0.8139)
        ("V3", 0.82531, "submission_3.csv",                            1, "V3 RoBERTa run 1 · val F1 0.8139",        ""),
        ("V3", 0.82929, "submission_2.csv",                            1, "V3 RoBERTa opt run 2 · val F1 0.8196",    ""),
        ("V3", 0.82837, "submission_1.csv",                            1, "V3 RoBERTa run 2 · val F1 0.8224",        ""),
        ("V3", 0.78976, "submission.csv",                              1, "V3 BERTweet · val F1 0.7834",             ""),
    ]

    # Anchor each Kaggle diamond near its version's circles on the X axis.
    series_kaggle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    version_x_anchor: dict[str, float] = {}
    for v in ("V1", "V2", "V3", "V4"):
        xs = [p["x"] for p in series_ok.get(v, [])] + [p["x"] for p in series_fail.get(v, [])]
        if xs:
            # Place diamonds at the *end* of each version's trial range (just past the last circle).
            version_x_anchor[v] = max(xs) + 1
        else:
            version_x_anchor[v] = len(all_sorted) + 1
    for v, score, fname, days_ago, matched_run, comment in KAGGLE_SCORES:
        # Spread Kaggle diamonds horizontally within the version's bucket so they don't all overlap.
        existing = len(series_kaggle[v])
        x = version_x_anchor[v] + existing * 0.8
        label = f"Kaggle public={score} · {fname} · {days_ago}d ago · ↪ {matched_run}"
        if comment:
            label += f" ({comment})"
        series_kaggle[v].append({"x": x, "y": round(score, 4), "label": label})

    # Sweep decisions for the running session
    decisions = tail_jsonl(SWEEP_DECISIONS, n=15)
    overall = load_json(OVERALL_BEST)

    # Active-session detection: most recently modified session dir
    active_session = None
    candidates = [d for d in glob(os.path.join(RUNS_DIR, "*")) if os.path.isdir(d)]
    if candidates:
        active_session = os.path.basename(max(candidates, key=os.path.getmtime))

    # Per-family success/fail counts for the active 60-min run
    current_run_trials = [t for t in trials if t["source"] == "agent4"]
    family_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "err": 0, "best_f1": None})
    for t in current_run_trials:
        bucket = family_breakdown[t["family"]]
        if t["success"]:
            bucket["ok"] += 1
            if isinstance(t["f1"], (int, float)) and (bucket["best_f1"] is None or t["f1"] > bucket["best_f1"]):
                bucket["best_f1"] = t["f1"]
        else:
            bucket["err"] += 1

    version_counts = Counter(t.get("version") for t in trials)
    body = []
    body.append("<div class='row'>")
    body.append(f"<div class='card neutral'><h3>Trials total</h3><div class='value'>{total}</div></div>")
    body.append(f"<div class='card good'><h3>Successful</h3><div class='value'>{len(successful)}</div></div>")
    body.append(f"<div class='card bad'><h3>Failed</h3><div class='value'>{len(failed)}</div></div>")
    body.append(f"<div class='card bad'><h3>Error rate</h3><div class='value'>{error_rate:.1f}%</div></div>")
    if best:
        body.append(f"<div class='card good'><h3>Best F1</h3><div class='value'>{best['f1']:.4f}</div><div class='small'>{best['version']} · {best['family']} · {best['session']}</div></div>")
    body.append("</div>")

    # Per-version stat strip
    body.append("<div class='row'>")
    for v in ("V1", "V2", "V3", "V4"):
        n = version_counts.get(v, 0)
        body.append(f"<div class='card neutral'><h3>{v} trials</h3><div class='value'>{n}</div></div>")
    body.append("</div>")

    body.append(f"<p class='small'>Active session (most recently modified): <b>{active_session or '—'}</b>. Page auto-refreshes every 10 s.</p>")

    body.append("<div class='chart-card'>")
    body.append("<h3 style='margin-top:0'>F1 across all versions — V1 / V2 / V3 / V4</h3>")
    body.append("<p class='small'>"
                "<b>Color</b> encodes the agent version (V1 grey · V2 blue · V3 amber · V4 green). "
                "<b>Shape</b> encodes the score type: "
                "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:#2563eb;vertical-align:middle'></span> circle = validation F1 (80/20 split, from each version's JSON log), "
                "<span style='display:inline-block;width:11px;height:11px;background:#fff;border:2px solid #16a34a;transform:rotate(45deg);vertical-align:middle'></span> diamond = official Kaggle public-leaderboard score, "
                "<span style='color:#dc2626;font-weight:700'>✗</span> = failure (plotted at y = −0.05). "
                "Kaggle diamonds are transcribed from the user's Kaggle submission history.</p>")
    body.append("<div style='height:420px'><canvas id='f1chart'></canvas></div>")
    body.append("</div>")

    # Kaggle public score → agent run matching (transcribed from Niccolò's Kaggle account)
    body.append("<h3 style='margin-top:28px'>Kaggle public scores — matched to agent run</h3>")
    body.append("<p class='small'>Each official Kaggle submission below is matched to the agent run that "
                "wrote the submission CSV, using submission date, filename, and Kaggle-vs-validation F1 "
                "proximity. The 7-day-ago entry has an exact date match to the May 7 V3 sweep.</p>")
    body.append("<table><tr><th>Version</th><th>Kaggle file</th><th>Public score</th>"
                "<th>Days ago</th><th>Matched agent run</th><th>Note</th></tr>")
    for v, score, fname, days_ago, matched_run, comment in KAGGLE_SCORES:
        body.append(
            f"<tr><td><b style='color:{ {'V1':'#9ca3af','V2':'#2563eb','V3':'#f59e0b','V4':'#16a34a'}.get(v,'#666') }'>{v}</b></td>"
            f"<td class='small'><code>{fname}</code></td>"
            f"<td><b>{score:.4f}</b></td>"
            f"<td class='small'>{days_ago}d</td>"
            f"<td class='small'>{matched_run}</td>"
            f"<td class='small'>{comment}</td></tr>"
        )
    body.append("</table>")

    # Family breakdown table
    body.append("<h3 style='margin-top:28px'>Per-family results — Agent_4 sessions</h3>")
    body.append("<table><tr><th>Family</th><th>Successful</th><th>Failed</th><th>Best F1</th></tr>")
    for fam, bucket in sorted(family_breakdown.items(), key=lambda kv: -(kv[1]["best_f1"] or 0)):
        best_f1_text = f"{bucket['best_f1']:.4f}" if bucket["best_f1"] is not None else "—"
        body.append(
            f"<tr><td><b>{fam}</b></td>"
            f"<td><span class='pill ok'>{bucket['ok']}</span></td>"
            f"<td><span class='pill err'>{bucket['err']}</span></td>"
            f"<td>{best_f1_text}</td></tr>"
        )
    body.append("</table>")

    # Recent sweep decisions
    body.append("<h3 style='margin-top:28px'>Recent sweep-planner decisions</h3>")
    if not decisions:
        body.append("<p class='small'>No decisions logged yet for the current session.</p>")
    else:
        body.append("<table><tr><th>Time</th><th>Action</th><th>Family</th><th>Reason</th><th>Time left</th></tr>")
        for d in reversed(decisions):
            cls = "ok" if d.get("action") == "try_family" else ("warn" if d.get("action") == "skip_family_permanently" else "err")
            body.append(
                f"<tr><td class='small'>{d.get('timestamp', '')}</td>"
                f"<td><span class='pill {cls}'>{d.get('action', '')}</span></td>"
                f"<td>{d.get('family_key') or '—'}</td>"
                f"<td class='small'>{(d.get('reason') or '')[:160]}</td>"
                f"<td class='small'>{d.get('time_remaining_seconds', '')}s</td></tr>"
            )
        body.append("</table>")

    if overall:
        body.append("<h3 style='margin-top:28px'>Last completed full-run result (overall_best.json)</h3>")
        body.append(
            f"<p>Best family: <b>{overall.get('best_family', '?')}</b>, "
            f"run {overall.get('best_run_index', '?')}, "
            f"metrics {overall.get('best_metrics', {})}. "
            f"Time elapsed: {overall.get('time_elapsed_seconds', '?')}s.</p>"
        )

    # Most recent 25 Agent_4 trials with link to the prompt chain
    agent4_trials = [t for t in trials if t["source"] == "agent4" and t.get("run_dir")]
    if agent4_trials:
        body.append("<h3 style='margin-top:28px'>Recent Agent_4 trials — click to see the prompt that produced each result</h3>")
        body.append("<table><tr><th>Session</th><th>Family</th><th>Run</th><th>Status</th><th>F1</th><th>Outcome</th><th>Prompts</th></tr>")
        for t in agent4_trials[-25:][::-1]:
            status = "ok" if t["success"] else "err"
            f1_txt = f"{t['f1']:.4f}" if isinstance(t["f1"], (int, float)) else "—"
            body.append(
                f"<tr><td class='small'>{t['session']}</td>"
                f"<td>{t['family']}</td>"
                f"<td>{t['run_index']}</td>"
                f"<td><span class='pill {status}'>{'success' if t['success'] else 'fail'}</span></td>"
                f"<td>{f1_txt}</td>"
                f"<td class='small'>{t['outcome']}</td>"
                f"<td class='small'><a href='/trial/{t['session']}/run_{t['run_index']:03d}'>view prompts ↗</a></td></tr>"
            )
        body.append("</table>")

    # Embed chart data as JS — three series per version: ok (circle), kaggle (diamond), fail (red X).
    ok_payload   = {v: [{"x": p["x"], "y": p["y"]} for p in pts] for v, pts in series_ok.items()}
    ok_labels    = {v: [p["label"] for p in pts] for v, pts in series_ok.items()}
    fail_payload = {v: [{"x": p["x"], "y": p["y"]} for p in pts] for v, pts in series_fail.items()}
    fail_labels  = {v: [p["label"] for p in pts] for v, pts in series_fail.items()}
    kag_payload  = {v: [{"x": p["x"], "y": p["y"]} for p in pts] for v, pts in series_kaggle.items()}
    kag_labels   = {v: [p["label"] for p in pts] for v, pts in series_kaggle.items()}
    body.append(f"""
<script>
const OK_SERIES     = {json.dumps(ok_payload)};
const OK_LABELS     = {json.dumps(ok_labels)};
const FAIL_SERIES   = {json.dumps(fail_payload)};
const FAIL_LABELS   = {json.dumps(fail_labels)};
const KAGGLE_SERIES = {json.dumps(kag_payload)};
const KAGGLE_LABELS = {json.dumps(kag_labels)};

// Vivid per-version palette. Shapes encode data type, colors encode version.
const COLORS = {{"V1": "#9ca3af", "V2": "#2563eb", "V3": "#f59e0b", "V4": "#16a34a"}};
const FAIL_COLOR = "#dc2626";

const datasets = [];
for (const v of ["V1", "V2", "V3", "V4"]) {{
    const c = COLORS[v] || "#666";
    if (OK_SERIES[v] && OK_SERIES[v].length) {{
        datasets.push({{
            label: v + " — validation F1 (◯)",
            data: OK_SERIES[v],
            showLine: false,
            borderColor: c,
            backgroundColor: c + "cc",
            pointRadius: 6,
            pointHoverRadius: 8,
            pointStyle: "circle",
            pointBorderWidth: 1.5,
        }});
    }}
    if (KAGGLE_SERIES[v] && KAGGLE_SERIES[v].length) {{
        datasets.push({{
            label: v + " — Kaggle public score (◆)",
            data: KAGGLE_SERIES[v],
            showLine: false,
            borderColor: c,
            backgroundColor: "#ffffff",
            pointRadius: 11,
            pointHoverRadius: 13,
            pointStyle: "rectRot",       // diamond
            pointBorderWidth: 3,
        }});
    }}
    if (FAIL_SERIES[v] && FAIL_SERIES[v].length) {{
        datasets.push({{
            label: v + " — failure (✗)",
            data: FAIL_SERIES[v],
            showLine: false,
            borderColor: FAIL_COLOR,
            backgroundColor: FAIL_COLOR,
            pointRadius: 7,
            pointHoverRadius: 9,
            pointStyle: "crossRot",
            pointBorderWidth: 2,
        }});
    }}
}}
new Chart(document.getElementById('f1chart'), {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        layout: {{ padding: {{ top: 8, right: 20, bottom: 8, left: 8 }} }},
        scales: {{
            y: {{
                min: -0.1, max: 0.9,
                grid: {{ color: "#eef2f7" }},
                title: {{
                    display: true,
                    text: "F1 score — circles = validation (80/20 split) · diamonds = Kaggle public · ✗ = failure",
                    font: {{ size: 12 }}
                }},
                ticks: {{
                    callback: v => v === -0.05 ? '✗ fail' : v.toFixed(2),
                }}
            }},
            x: {{
                grid: {{ color: "#eef2f7" }},
                title: {{ display: true, text: "trial # (chronological, grouped by version)" }}
            }}
        }},
        plugins: {{
            legend: {{
                position: "top",
                labels: {{ usePointStyle: true, pointStyle: "circle", padding: 12 }}
            }},
            tooltip: {{
                callbacks: {{
                    label: ctx => {{
                        const lab = ctx.dataset.label || "";
                        const v = lab.split(" ")[0];
                        const isFail = lab.includes("failure");
                        const isKag  = lab.includes("Kaggle");
                        const labels = isFail ? (FAIL_LABELS[v] || [])
                                       : isKag ? (KAGGLE_LABELS[v] || [])
                                              : (OK_LABELS[v] || []);
                        const txt = labels[ctx.dataIndex] || "";
                        return isFail ? txt : `${{txt}} · F1 = ${{ctx.parsed.y}}`;
                    }}
                }}
            }},
            title: {{
                display: true,
                text: "Validation F1 by agent version — circles = 80/20 split, diamonds = Kaggle public"
            }}
        }}
    }}
}});
</script>
""")

    return render("\n".join(body))


@app.route("/current")
def current_input():
    """Show the spec/prompt/code currently being processed by the agent."""
    session_dir = latest_session_dir()
    if not session_dir:
        return render("<p>No sessions found yet.</p>")
    run_dir = latest_run_dir(session_dir)

    spec_path = os.path.join(run_dir, "spec.json") if run_dir else None
    prompt_path = os.path.join(run_dir, "prompt.txt") if run_dir else None
    train_path = os.path.join(run_dir, "train.py") if run_dir else None
    log_path = os.path.join(run_dir, "run.log") if run_dir else None

    spec = load_json(spec_path) if spec_path else None
    prompt = load_text(prompt_path) if prompt_path else ""
    train_code = load_text(train_path, max_bytes=80_000) if train_path else ""
    log_text = load_text(log_path, max_bytes=40_000) if log_path else ""

    decisions = tail_jsonl(SWEEP_DECISIONS, n=1)
    last_decision = decisions[-1] if decisions else None

    body = []
    body.append(f"<h2>Current session: <code>{os.path.basename(session_dir)}</code></h2>")
    if run_dir:
        body.append(f"<p class='small'>Latest run dir: <code>{os.path.basename(run_dir)}</code></p>")

    if last_decision:
        body.append("<h3>Latest sweep-planner decision</h3>")
        body.append(f"<pre class='code'>{json.dumps(last_decision, indent=2)}</pre>")

    if spec:
        body.append("<h3>Current spec.json</h3>")
        body.append(f"<pre class='code'>{json.dumps(spec, indent=2)}</pre>")

    if prompt:
        body.append("<h3>Prompt sent to code-gen LLM</h3>")
        body.append(f"<pre class='code'>{prompt[:8000]}</pre>")

    if log_text:
        body.append("<h3>run.log (tail)</h3>")
        body.append(f"<pre class='code'>{log_text[-8000:]}</pre>")

    if train_code:
        body.append("<h3>Generated training script</h3>")
        body.append(f"<pre class='code'>{train_code[:8000]}</pre>")

    return render("\n".join(body))


@app.route("/errors")
def errors_page():
    trials = gather_trials()
    failed = [t for t in trials if not t["success"]]
    by_outcome = Counter(t["outcome"] for t in failed)
    by_family = Counter(t["family"] for t in failed)

    total = len(trials)
    error_rate = (len(failed) / total * 100) if total else 0.0

    body = []
    body.append("<div class='row'>")
    body.append(f"<div class='card bad'><h3>Failed trials</h3><div class='value'>{len(failed)}</div></div>")
    body.append(f"<div class='card bad'><h3>Error rate</h3><div class='value'>{error_rate:.1f}%</div></div>")
    body.append(f"<div class='card neutral'><h3>Trials total</h3><div class='value'>{total}</div></div>")
    body.append("</div>")

    body.append("<div class='chart-card'><canvas id='outcomeChart' height='110'></canvas></div>")

    body.append("<h3 style='margin-top:24px'>Outcome breakdown</h3>")
    body.append("<table><tr><th>Outcome</th><th>Count</th></tr>")
    for outcome, count in by_outcome.most_common():
        body.append(f"<tr><td><span class='pill err'>{outcome}</span></td><td>{count}</td></tr>")
    body.append("</table>")

    body.append("<h3 style='margin-top:24px'>Failures by family</h3>")
    body.append("<table><tr><th>Family</th><th>Failed trials</th></tr>")
    for fam, count in by_family.most_common():
        body.append(f"<tr><td><b>{fam}</b></td><td>{count}</td></tr>")
    body.append("</table>")

    body.append("<h3 style='margin-top:24px'>Last 30 failures — click to see prompt chain</h3>")
    body.append("<table><tr><th>Session</th><th>Family</th><th>Run #</th><th>Outcome</th><th>Error tail</th><th>Prompts</th></tr>")
    for t in failed[-30:][::-1]:
        link = (f"<a href='/trial/{t['session']}/run_{t['run_index']:03d}'>view prompts ↗</a>"
                if t.get("run_dir") else "—")
        body.append(
            f"<tr><td class='small'>{t['session']}</td>"
            f"<td>{t['family']}</td>"
            f"<td>{t['run_index']}</td>"
            f"<td><span class='pill err'>{t['outcome']}</span></td>"
            f"<td class='small'>{(t.get('error_summary') or '')[:120]}</td>"
            f"<td class='small'>{link}</td></tr>"
        )
    body.append("</table>")

    outcome_data = dict(by_outcome)
    body.append(f"""
<script>
new Chart(document.getElementById('outcomeChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(list(outcome_data.keys()))},
        datasets: [{{
            label: 'Failures by outcome',
            data: {json.dumps(list(outcome_data.values()))},
            backgroundColor: '#c0392b'
        }}]
    }},
    options: {{ plugins: {{ legend: {{display: false}} }}, scales: {{ y: {{beginAtZero: true}} }} }}
}});
</script>
""")

    return render("\n".join(body))


@app.route("/trial/<session>/<run_name>")
def trial_detail(session: str, run_name: str):
    """Show the full prompt-chain for a specific trial (success or failure)."""
    # Reject path traversal
    if "/" in session or "\\" in session or "/" in run_name or "\\" in run_name:
        abort(400)
    run_dir = os.path.join(RUNS_DIR, session, run_name)
    if not os.path.isdir(run_dir):
        abort(404, description=f"No such run dir: {session}/{run_name}")

    # Locate which spec-gen variant ran (initial uses spec_prompt, revisit uses search_prompt)
    spec_prompt = load_text(os.path.join(run_dir, "spec_prompt.txt"))
    if not spec_prompt:
        spec_prompt = load_text(os.path.join(run_dir, "search_prompt.txt"))
        spec_label = "Spec-gen prompt (search / revisit)"
    else:
        spec_label = "Spec-gen prompt (initial)"
    spec_response = load_text(os.path.join(run_dir, "spec_response.txt"))
    if not spec_response:
        spec_response = load_text(os.path.join(run_dir, "search_response.txt"))

    spec = load_json(os.path.join(run_dir, "spec.json"))
    code_prompt = load_text(os.path.join(run_dir, "prompt.txt"))
    code_response = load_text(os.path.join(run_dir, "generation_response.txt"))
    train_py = load_text(os.path.join(run_dir, "train.py"), max_bytes=80_000)
    run_log = load_text(os.path.join(run_dir, "run.log"), max_bytes=40_000)
    metrics = load_json(os.path.join(run_dir, "metrics.json"))

    # Repair attempts (numbered repair_attempt_1.json, _2.json, ...)
    repair_paths = sorted(glob(os.path.join(run_dir, "repair_attempt_*.json")))
    repairs: list[dict[str, Any]] = []
    for rp in repair_paths:
        repairs.append({
            "attempt": os.path.basename(rp).replace("repair_attempt_", "").replace(".json", ""),
            "raw": load_text(rp, max_bytes=20_000),
        })

    # Outcome look-up so we can flag success/failure at the top
    outcome = None
    trial_info: dict[str, Any] = {}
    for t in gather_trials():
        if t["session"] == session and isinstance(t["run_index"], int) and \
                f"run_{t['run_index']:03d}" == run_name:
            outcome = t["outcome"]
            trial_info = t
            break

    body = []
    body.append(f"<h2>Trial: <code>{session} / {run_name}</code></h2>")
    body.append("<p><a href='/'>← back to overview</a> · <a href='/errors'>errors</a></p>")

    pill_cls = "ok" if trial_info.get("success") else "err"
    status_text = "success" if trial_info.get("success") else (outcome or "failure")
    f1_text = f"{trial_info['f1']:.4f}" if isinstance(trial_info.get("f1"), (int, float)) else "—"
    body.append("<div class='row'>")
    body.append(f"<div class='card neutral'><h3>Family</h3><div class='value' style='font-size:18px'>{trial_info.get('family', '?')}</div></div>")
    body.append(f"<div class='card {'good' if trial_info.get('success') else 'bad'}'><h3>Status</h3><div class='value' style='font-size:18px'><span class='pill {pill_cls}'>{status_text}</span></div></div>")
    body.append(f"<div class='card good'><h3>F1</h3><div class='value' style='font-size:22px'>{f1_text}</div></div>")
    if trial_info.get("repair_attempts"):
        body.append(f"<div class='card bad'><h3>Repair attempts</h3><div class='value' style='font-size:22px'>{trial_info['repair_attempts']}</div></div>")
    if trial_info.get("wall_seconds"):
        body.append(f"<div class='card neutral'><h3>Wall seconds</h3><div class='value' style='font-size:22px'>{trial_info['wall_seconds']}</div></div>")
    body.append("</div>")

    if trial_info.get("error_summary"):
        body.append(f"<p><b>Error tail:</b> <code>{trial_info['error_summary']}</code></p>")

    # 1. Spec prompt
    if spec_prompt:
        body.append(f"<h3>1. {spec_label}</h3>")
        body.append(f"<pre class='code'>{spec_prompt[:12000]}</pre>")
    if spec_response:
        body.append("<h3>2. Spec-gen LLM response</h3>")
        body.append(f"<pre class='code'>{spec_response[:8000]}</pre>")

    # 2. Validated spec
    if spec:
        body.append("<h3>3. Validated spec.json</h3>")
        body.append(f"<pre class='code'>{json.dumps(spec, indent=2)}</pre>")

    # 3. Code-gen prompt + response
    if code_prompt:
        body.append("<h3>4. Code-gen prompt</h3>")
        body.append(f"<pre class='code'>{code_prompt[:12000]}</pre>")
    if code_response:
        body.append("<h3>5. Code-gen LLM response</h3>")
        body.append(f"<pre class='code'>{code_response[:12000]}</pre>")

    # 4. Generated code
    if train_py:
        body.append("<h3>6. Generated training script (post-repair if any)</h3>")
        body.append(f"<pre class='code'>{train_py[:20000]}</pre>")

    # 5. Repair attempts — these are the prompts/responses produced when a trial failed.
    if repairs:
        body.append(f"<h3>7. Surgical repair attempts ({len(repairs)})</h3>")
        for rep in repairs:
            body.append(f"<h4>Repair attempt {rep['attempt']}</h4>")
            body.append(f"<pre class='code'>{rep['raw']}</pre>")

    # 6. run.log
    if run_log:
        body.append("<h3>8. run.log (execution + analysis)</h3>")
        body.append(f"<pre class='code'>{run_log[-10000:]}</pre>")

    if metrics:
        body.append("<h3>9. metrics.json</h3>")
        body.append(f"<pre class='code'>{json.dumps(metrics, indent=2)}</pre>")

    return render("\n".join(body))


@app.route("/architectures")
def architectures():
    """One page with all four architecture diagrams + a short description of each version."""
    versions = [
        {
            "id": "v1",
            "title": "V1 — Single-architecture LLM Baseline",
            "subtitle": "src/agents/v1_simple.py (Keras embedding + LSTM)",
            "summary": (
                "LLM is asked for a single experiment dict (clamped to a 4-key search space: "
                "model_type ∈ {avg_embed, lstm}, vocab_size ∈ {5k, 10k}, seq_length ∈ {50, 100}, "
                "embed_dim ∈ {32, 64}). One script trains and reports F1; results appended to a JSONL log "
                "and fed back into the next prompt as history."
            ),
        },
        {
            "id": "v2",
            "title": "V2 — Single-file Autonomous Multi-architecture Agent",
            "subtitle": "Agent_V2/agent_fully_autonomous.py",
            "summary": (
                "One Python file drives a full experiment loop across REQUIRED_ARCHITECTURES. "
                "LLM proposes a complete training script; preflight + syntax checks run before execution; "
                "failures trigger a single LLM repair pass. ExperimentMemory tracks rolling history, best F1, "
                "and a plateau detector. Log is a flat experiment_log.json."
            ),
        },
        {
            "id": "v3",
            "title": "V3 — Multi-family Modular Agent",
            "subtitle": "src/Agent_3/agent.py (sweep + opt + final retrain)",
            "summary": (
                "Refactored into a package: per-family hooks (BoW · BoW_advanced · CNN · LSTM · Embedding · "
                "RoBERTa · BERTweet), Jinja templates, surgical repair contract (≤8 attempts), tiered memory. "
                "Sweep walks all families on a 4k sample; top-2 architectures then enter an optimize phase on 10k rows; "
                "final retrain on full labeled data. Flat agent3_log.json."
            ),
        },
        {
            "id": "v4",
            "title": "V4 — LLM-driven Sweep Planner",
            "subtitle": "src/Agent_4/agent.py (this codebase)",
            "summary": (
                "Sweep order is no longer a hardcoded list — an LLM planner reads the per-family state table "
                "every step and chooses try_family / skip_family_permanently / stop. Sweep ends at a fixed "
                "40-min wall-clock boundary. 2k-row fixed sample is shared across sweep/opt/final retrain. "
                "New artifact: sweep_decisions.jsonl logs every planner decision with prompt + raw response."
            ),
        },
    ]
    body = ["<h2>Agent versions</h2>",
            "<p class='small'>PNGs are stored under <code>src/Agent_4/docs/</code> and rendered from the "
            ".dot sources next to them.</p>"]
    for v in versions:
        body.append("<div class='chart-card' style='margin-bottom:24px'>")
        body.append(f"<h3 style='margin-top:0'>{v['title']}</h3>")
        body.append(f"<p class='small'>{v['subtitle']}</p>")
        body.append(f"<p>{v['summary']}</p>")
        body.append(f"<img src='/docs/architecture_{v['id']}.png' alt='{v['title']}' "
                    f"style='max-width:100%; border:1px solid #eee; border-radius:6px'>")
        body.append("</div>")
    return render("\n".join(body))


@app.route("/docs/<path:filename>")
def serve_docs(filename: str):
    return send_from_directory(DOCS_DIR, filename)


# Where each version's Kaggle-ready submission CSV lives (or "not produced" if missing).
SUBMISSION_PATHS = {
    "V1": r"C:\Users\shoxx\Downloads\apa-disaster-tweets-agent\submissions",
    "V2": r"C:\Users\shoxx\Downloads\Agent_V2\submissions",
    "V3": os.path.join(PROJECT_ROOT, "submissions"),
    "V4": os.path.join(AGENT_ROOT, "submissions"),
}


def submission_summary(folder: str) -> dict[str, Any]:
    """Return {csv_count, latest_csv, latest_rows, class_dist} for a submissions folder."""
    if not os.path.isdir(folder):
        return {"exists": False}
    csvs = [p for p in glob(os.path.join(folder, "*.csv"))]
    if not csvs:
        return {"exists": True, "csv_count": 0}
    latest = max(csvs, key=os.path.getmtime)
    rows, ones, zeros = 0, 0, 0
    try:
        with open(latest, "r", encoding="utf-8", errors="replace") as fh:
            next(fh, None)  # skip header
            for line in fh:
                rows += 1
                tail = line.rstrip().rsplit(",", 1)[-1]
                if tail == "1":
                    ones += 1
                elif tail == "0":
                    zeros += 1
    except Exception:
        pass
    return {
        "exists": True,
        "csv_count": len(csvs),
        "latest_csv": os.path.basename(latest),
        "latest_path": latest,
        "latest_rows": rows,
        "ones": ones,
        "zeros": zeros,
        "mtime": datetime.fromtimestamp(os.path.getmtime(latest)).strftime("%Y-%m-%d %H:%M"),
    }


@app.route("/comparison")
def comparison():
    """Per-version table: best validation F1, failures, submission CSV, Kaggle score."""
    trials = gather_trials()
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trials:
        by_version[t.get("version", "?")].append(t)

    body = [
        "<h2>Version comparison</h2>",
        "<p class='small'>Validation F1 = recorded in each version's JSON log. "
        "Kaggle public score = pulled from agent log only if AGENT3_AUTO_SUBMIT_KAGGLE was set when the run "
        "happened. None of the V1–V4 runs in this project recorded a Kaggle response, so that column is "
        "marked 'not recorded' where missing.</p>",
        "<table>",
        "<tr><th>Version</th><th>Total trials</th><th>Succeeded</th><th>Failed</th>"
        "<th>Best validation F1</th><th>Best F1 — who</th>"
        "<th>Failure outcomes</th>"
        "<th>Submission CSV</th>"
        "<th>Kaggle public score</th></tr>",
    ]

    for v in ("V1", "V2", "V3", "V4"):
        rows = by_version.get(v, [])
        succ = [t for t in rows if t.get("success") and isinstance(t.get("f1"), (int, float))]
        fail = [t for t in rows if not t.get("success")]
        if succ:
            best = max(succ, key=lambda t: t["f1"])
            best_f1 = f"{best['f1']:.4f}"
            best_who = f"{best.get('family', '?')} · {best.get('session', '?')}"
        else:
            best_f1 = "—"
            best_who = "no successful trial in log"
        fail_counts = Counter(t.get("outcome", "unknown") for t in fail)
        fail_text = ", ".join(f"{k}={c}" for k, c in fail_counts.most_common()) or "—"

        sub = submission_summary(SUBMISSION_PATHS.get(v, ""))
        if not sub.get("exists"):
            sub_text = "<span class='small'>folder missing</span>"
        elif sub.get("csv_count", 0) == 0:
            sub_text = "<span class='pill err'>no CSV produced</span>"
        else:
            sub_text = (f"<b>{sub['csv_count']}</b> CSV(s); latest: "
                        f"<code>{sub['latest_csv']}</code> "
                        f"<span class='small'>({sub['latest_rows']} rows · "
                        f"0:{sub['zeros']} / 1:{sub['ones']} · {sub['mtime']})</span>")

        # Kaggle score lookup — V4 stores it under overall_best.kaggle_submission if AGENT3_AUTO_SUBMIT_KAGGLE was on.
        kaggle_text = "<span class='small'>not recorded</span>"
        if v == "V4":
            overall = load_json(OVERALL_BEST) or {}
            ks = overall.get("kaggle_submission")
            if isinstance(ks, dict) and ks.get("submitted"):
                kaggle_text = (f"public={ks.get('public_score', '?')} · "
                               f"private={ks.get('private_score', '?')} · "
                               f"status={ks.get('status', '?')}")

        body.append(
            f"<tr><td><b>{v}</b></td>"
            f"<td>{len(rows)}</td>"
            f"<td><span class='pill ok'>{len(succ)}</span></td>"
            f"<td><span class='pill err'>{len(fail)}</span></td>"
            f"<td><b>{best_f1}</b></td>"
            f"<td class='small'>{best_who}</td>"
            f"<td class='small'>{fail_text}</td>"
            f"<td class='small'>{sub_text}</td>"
            f"<td class='small'>{kaggle_text}</td></tr>"
        )
    body.append("</table>")

    # Per-version successful-trial detail
    body.append("<h3 style='margin-top:28px'>Successful trials per version</h3>")
    body.append("<table><tr><th>Version</th><th>Family / arch</th><th>Run / session</th><th>F1</th><th>Accuracy</th></tr>")
    for v in ("V1", "V2", "V3", "V4"):
        for t in sorted(
            [x for x in by_version.get(v, []) if x.get("success") and isinstance(x.get("f1"), (int, float))],
            key=lambda x: -x["f1"],
        ):
            acc = t.get("accuracy")
            acc_text = f"{acc:.4f}" if isinstance(acc, (int, float)) else "—"
            body.append(
                f"<tr><td><b>{v}</b></td>"
                f"<td>{t.get('family', '?')}</td>"
                f"<td class='small'>{t.get('session', '?')}</td>"
                f"<td><b>{t['f1']:.4f}</b></td>"
                f"<td>{acc_text}</td></tr>"
            )
    body.append("</table>")

    # Per-version failures
    body.append("<h3 style='margin-top:28px'>Failures per version</h3>")
    body.append("<table><tr><th>Version</th><th>Family / arch</th><th>Outcome</th><th>Error tail</th></tr>")
    for v in ("V1", "V2", "V3", "V4"):
        for t in [x for x in by_version.get(v, []) if not x.get("success")]:
            link = ""
            if v == "V4" and t.get("run_dir"):
                link = f" <a href='/trial/{t['session']}/run_{t['run_index']:03d}'>↗</a>"
            body.append(
                f"<tr><td><b>{v}</b></td>"
                f"<td>{t.get('family', '?')}</td>"
                f"<td><span class='pill err'>{t.get('outcome', 'unknown')}</span>{link}</td>"
                f"<td class='small'>{(t.get('error_summary') or '')[:140]}</td></tr>"
            )
    body.append("</table>")

    return render("\n".join(body))


@app.route("/api/trials")
def api_trials():
    return jsonify(gather_trials())


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
