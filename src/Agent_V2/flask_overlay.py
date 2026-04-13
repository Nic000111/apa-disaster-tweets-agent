import json
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
AGENT_SCRIPT = Path(__file__).resolve().with_name("agent_fully_autonomous_test.py")
LOG_CANDIDATES = [
    ROOT_DIR / "experiment_log.json",
    Path(__file__).resolve().with_name("experiment_log.json"),
]

PROCESS_LOCK = threading.Lock()
RUN_PROCESS: subprocess.Popen | None = None
RUN_COMMAND: list[str] = []
RUN_STARTED_AT: float | None = None
STDOUT_BUFFER: deque[str] = deque(maxlen=300)
STDERR_BUFFER: deque[str] = deque(maxlen=300)


def _find_log_file() -> Path | None:
    for candidate in LOG_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _read_log_rows() -> list[dict]:
    log_file = _find_log_file()
    if not log_file:
        return []

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        return []
    except (OSError, json.JSONDecodeError):
        return []


def _buffer_stream(stream, target: deque[str], prefix: str) -> None:
    try:
        for line in iter(stream.readline, ""):
            text = line.rstrip("\n")
            if text:
                target.append(f"[{prefix}] {text}")
    finally:
        stream.close()


def _is_running() -> bool:
    return RUN_PROCESS is not None and RUN_PROCESS.poll() is None


def _status_payload() -> dict:
    running = _is_running()
    return_code = RUN_PROCESS.poll() if RUN_PROCESS else None
    return {
        "running": running,
        "pid": RUN_PROCESS.pid if RUN_PROCESS else None,
        "return_code": return_code,
        "started_at": RUN_STARTED_AT,
        "elapsed_sec": round(time.time() - RUN_STARTED_AT, 2) if (RUN_STARTED_AT and running) else None,
        "command": RUN_COMMAND,
        "stdout_tail": list(STDOUT_BUFFER)[-40:],
        "stderr_tail": list(STDERR_BUFFER)[-40:],
    }


@app.get("/")
def index():
    return render_template("overlay.html")


@app.get("/api/summary")
def summary():
    rows = _read_log_rows()
    successful = [r for r in rows if r.get("success")]
    scores = [
        (r.get("metrics") or {}).get("f1")
        for r in successful
        if isinstance((r.get("metrics") or {}).get("f1"), (int, float))
    ]

    recent = []
    for row in rows[-12:][::-1]:
        metrics = row.get("metrics") or {}
        recent.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "architecture": row.get("architecture"),
                "success": bool(row.get("success")),
                "f1": metrics.get("f1"),
                "accuracy": metrics.get("accuracy"),
                "timestamp": row.get("timestamp"),
            }
        )

    return jsonify(
        {
            "total_runs": len(rows),
            "successful_runs": len(successful),
            "failed_runs": len(rows) - len(successful),
            "best_f1": max(scores) if scores else None,
            "log_file": str(_find_log_file()) if _find_log_file() else None,
            "recent_runs": recent,
        }
    )


@app.get("/api/status")
def status():
    with PROCESS_LOCK:
        return jsonify(_status_payload())


@app.post("/api/start")
def start_agent():
    payload = request.get_json(silent=True) or {}
    model = str(payload.get("model", "gemma4:latest")).strip() or "gemma4:latest"
    max_iter = int(payload.get("max_iter", 2))
    fresh = bool(payload.get("fresh", False))

    command = [
        sys.executable,
        str(AGENT_SCRIPT),
        "--model",
        model,
        "--max-iter",
        str(max_iter),
    ]
    if fresh:
        command.append("--fresh")

    with PROCESS_LOCK:
        if _is_running():
            return jsonify({"ok": False, "message": "Agent is already running."}), 409

        STDOUT_BUFFER.clear()
        STDERR_BUFFER.clear()

        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        global RUN_PROCESS, RUN_COMMAND, RUN_STARTED_AT
        RUN_PROCESS = process
        RUN_COMMAND = command
        RUN_STARTED_AT = time.time()

        threading.Thread(target=_buffer_stream, args=(process.stdout, STDOUT_BUFFER, "OUT"), daemon=True).start()
        threading.Thread(target=_buffer_stream, args=(process.stderr, STDERR_BUFFER, "ERR"), daemon=True).start()

        return jsonify({"ok": True, "message": "Autonomous agent started.", "pid": process.pid})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
