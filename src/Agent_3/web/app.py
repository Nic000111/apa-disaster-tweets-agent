"""Flask overlay for Agent_3: launch the agent, browse prompts, view F1 graph."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request


WEB_DIR = Path(__file__).resolve().parent
AGENT_DIR = WEB_DIR.parent
REPO_ROOT = AGENT_DIR.parent.parent
RUNS_DIR = AGENT_DIR / "runs"
WEB_LOGS_DIR = WEB_DIR / "logs"
WEB_LOGS_DIR.mkdir(exist_ok=True)

FAMILIES = ["bow", "bow_advanced", "cnn", "embedding_dl", "lstm", "transformer", "roberta", "bertweet"]

app = Flask(__name__, template_folder=str(WEB_DIR / "templates"))

_state_lock = threading.Lock()
_state: dict = {
    "process": None,
    "log_path": None,
    "started_at": None,
    "command": None,
    "log_handle": None,
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


@app.route("/")
def index():
    return render_template("index.html", families=FAMILIES)


@app.route("/api/sessions")
def list_sessions():
    sessions = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not d.is_dir() or d.name == "__pycache__":
                continue
            summary = _read_json(d / "summary.json")
            run_count = sum(1 for c in d.iterdir() if c.is_dir() and c.name.startswith("run_"))
            sessions.append({
                "name": d.name,
                "family": summary.get("family", ""),
                "phase": summary.get("phase", ""),
                "best_run_index": summary.get("best_run_index"),
                "best_metrics": summary.get("best_metrics", {}),
                "run_count": run_count,
                "has_summary": (d / "summary.json").exists(),
            })
    return jsonify(sessions)


@app.route("/api/sessions/<session>")
def session_detail(session: str):
    session_dir = RUNS_DIR / session
    if not session_dir.exists() or not session_dir.is_dir():
        abort(404)
    summary = _read_json(session_dir / "summary.json")
    runs = []
    for run_dir in sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("run_")):
        metrics_blob = _read_json(run_dir / "metrics.json")
        spec = _read_json(run_dir / "spec.json")
        metrics = metrics_blob.get("metrics", {}) if isinstance(metrics_blob, dict) else {}
        success = bool(metrics_blob.get("success")) if isinstance(metrics_blob, dict) else False
        runs.append({
            "run": run_dir.name,
            "success": success,
            "timed_out": bool(metrics_blob.get("timed_out", False)),
            "metrics": metrics,
            "spec": spec,
            "prompt": _read_text(run_dir / "prompt.txt"),
            "search_prompt": _read_text(run_dir / "search_prompt.txt"),
            "generation_response": _read_text(run_dir / "generation_response.txt"),
            "search_response": _read_text(run_dir / "search_response.txt"),
        })
    return jsonify({"session": session, "summary": summary, "runs": runs})


@app.route("/api/launch", methods=["POST"])
def launch():
    body = request.get_json(silent=True) or {}
    family = (body.get("family") or "").strip() or None
    try:
        minutes = max(1, int(body.get("minutes") or 80))
    except (TypeError, ValueError):
        minutes = 80
    no_winner = bool(body.get("no_winner_optimization"))
    fresh = bool(body.get("fresh"))
    max_runs = body.get("max_runs")
    try:
        max_runs = int(max_runs) if max_runs not in (None, "") else None
    except (TypeError, ValueError):
        max_runs = None

    cmd = [sys.executable, "-u", str(AGENT_DIR / "agent.py"), "--time-budget-minutes", str(minutes)]
    if family:
        cmd += ["--family", family]
    if no_winner:
        cmd += ["--no-winner-optimization"]
    if fresh:
        cmd += ["--fresh"]
    if max_runs is not None:
        cmd += ["--max-runs", str(max_runs)]

    with _state_lock:
        proc = _state["process"]
        if proc and proc.poll() is None:
            return jsonify({"error": "Agent is already running."}), 409
        if _state["log_handle"]:
            try:
                _state["log_handle"].close()
            except OSError:
                pass
            _state["log_handle"] = None

        log_name = f"agent_{time.strftime('%Y%m%d_%H%M%S')}.log"
        log_path = WEB_LOGS_DIR / log_name
        log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            new_proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except OSError as exc:
            log_handle.close()
            return jsonify({"error": f"Failed to start agent: {exc}"}), 500

        _state["process"] = new_proc
        _state["log_path"] = str(log_path)
        _state["started_at"] = time.time()
        _state["command"] = cmd
        _state["log_handle"] = log_handle

    return jsonify({"started": True, "log_path": str(log_path), "command": cmd})


@app.route("/api/status")
def status():
    with _state_lock:
        proc = _state["process"]
        running = bool(proc and proc.poll() is None)
        exit_code = None
        if proc and not running:
            exit_code = proc.returncode
        return jsonify({
            "running": running,
            "exit_code": exit_code,
            "log_path": _state["log_path"],
            "started_at": _state["started_at"],
            "command": _state["command"],
        })


@app.route("/api/log")
def get_log():
    offset = request.args.get("offset", default=0, type=int) or 0
    with _state_lock:
        log_path = _state["log_path"]
    if not log_path or not os.path.exists(log_path):
        return jsonify({"text": "", "size": 0, "offset": 0})
    size = os.path.getsize(log_path)
    if offset > size or offset < 0:
        offset = 0
    with open(log_path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    return jsonify({
        "text": chunk.decode("utf-8", errors="replace"),
        "size": size,
        "offset": size,
    })


@app.route("/api/stop", methods=["POST"])
def stop():
    with _state_lock:
        proc = _state["process"]
        if not proc or proc.poll() is not None:
            return jsonify({"stopped": False, "error": "No running agent."}), 400
        try:
            proc.terminate()
        except OSError as exc:
            return jsonify({"stopped": False, "error": str(exc)}), 500
    return jsonify({"stopped": True})


@app.route("/api/overall_best")
def overall_best():
    return jsonify(_read_json(RUNS_DIR / "overall_best.json"))


def main() -> None:
    port = int(os.environ.get("AGENT3_WEB_PORT", "5000"))
    host = os.environ.get("AGENT3_WEB_HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
