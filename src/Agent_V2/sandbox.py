"""
sandbox.py — Safe code execution with dry-run sanity check.

Matches the diagram:
  dry run (1 epoch, 200 samples) → pass → full EXECUTE → MONITOR (poll log)
                                  → fail → capture error, skip full run

The generated code runs in a subprocess so crashes can't kill the agent.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

DRY_RUN_TIMEOUT = 60    # seconds for the 1-epoch sanity check
FULL_RUN_TIMEOUT = 600  # seconds for the full training run (10 minutes max)
EXPERIMENT_DIR = "submissions"


def _write_temp_script(code: str, dry_run: bool) -> str:
    """Write code to a temp .py file. Inject dry-run flag if needed."""
    if dry_run:
        # Inject a _DRY_RUN = True flag at the top so the script can limit epochs/samples
        injection = (
            "import os as _os\n"
            "_os.environ['AGENT_DRY_RUN'] = '1'\n"
        )
        code = injection + code

    fd, path = tempfile.mkstemp(suffix=".py", prefix="agent_exp_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _parse_metrics(stdout: str) -> dict:
    """
    Look for the last line matching: METRICS: {"f1": 0.82, ...}
    Handles both single and double quotes.
    Returns empty dict if not found.
    """
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("METRICS:"):
            try:
                json_part = line[len("METRICS:"):].strip()
                # Replace single quotes with double quotes for valid JSON
                json_part = json_part.replace("'", '"')
                return json.loads(json_part)
            except json.JSONDecodeError:
                # Try extracting f1 with regex as fallback
                import re
                f1_match = re.search(r"f1['\"]?\s*:\s*([0-9.]+)", line)
                acc_match = re.search(r"accuracy['\"]?\s*:\s*([0-9.]+)", line)
                if f1_match:
                    return {
                        "f1": float(f1_match.group(1)),
                        "accuracy": float(acc_match.group(1)) if acc_match else 0.0,
                    }
    return {}


def run_experiment(code: str, name: str) -> dict:
    """
    Full pipeline:
      1. Dry run (1 epoch, 200 samples) — quick sanity check
      2. If dry run passes → full run
      3. Return result dict

    Result dict keys:
      success, timed_out, dry_run_failed, metrics, stdout, stderr
    """
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # ── Step 1: Dry run ──────────────────────────────────────────
    print(f"  [Sandbox] Dry run for '{name}'...")
    dry_path = _write_temp_script(code, dry_run=True)
    dry_result = _run_subprocess(dry_path, DRY_RUN_TIMEOUT)
    os.unlink(dry_path)

    if not dry_result["success"]:
        print(f"  [Sandbox] Dry run FAILED — skipping full run.")
        return {
            "success": False,
            "timed_out": dry_result["timed_out"],
            "dry_run_failed": True,
            "metrics": {},
            "stdout": dry_result["stdout"],
            "stderr": dry_result["stderr"],
        }

    print(f"  [Sandbox] Dry run passed. Starting full run...")

    # ── Step 2: Full run ─────────────────────────────────────────
    full_path = _write_temp_script(code, dry_run=False)
    full_result = _run_subprocess(full_path, FULL_RUN_TIMEOUT, monitor=True)
    os.unlink(full_path)

    metrics = _parse_metrics(full_result["stdout"])

    if not metrics and full_result["success"]:
        print("  [Sandbox] WARNING: Script ran but printed no METRICS line.")

    return {
        "success": full_result["success"],
        "timed_out": full_result["timed_out"],
        "dry_run_failed": False,
        "metrics": metrics,
        "stdout": full_result["stdout"],
        "stderr": full_result["stderr"],
    }


def _run_subprocess(script_path: str, timeout: int, monitor: bool = False) -> dict:
    """
    Run a Python script in a subprocess.
    If monitor=True, print a heartbeat dot every 30 seconds (MONITOR step from diagram).
    """
    start = time.time()
    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if monitor:
            # MONITOR: poll without calling LLM, just show progress
            stdout_lines = []
            stderr_lines = []
            last_heartbeat = start

            while True:
                elapsed = time.time() - start
                if elapsed > timeout:
                    proc.kill()
                    return {
                        "success": False,
                        "timed_out": True,
                        "stdout": "\n".join(stdout_lines),
                        "stderr": "TIMEOUT after {}s".format(timeout),
                    }

                # Print heartbeat every 30s
                if time.time() - last_heartbeat > 30:
                    print(f"  [Monitor] Still running... {int(elapsed)}s elapsed")
                    last_heartbeat = time.time()

                ret = proc.poll()
                if ret is not None:
                    out, err = proc.communicate()
                    stdout_lines.append(out)
                    stderr_lines.append(err)
                    return {
                        "success": ret == 0,
                        "timed_out": False,
                        "stdout": "\n".join(stdout_lines),
                        "stderr": "\n".join(stderr_lines),
                    }
                time.sleep(2)
        else:
            # Simple blocking call for dry runs
            stdout, stderr = proc.communicate(timeout=timeout)
            return {
                "success": proc.returncode == 0,
                "timed_out": False,
                "stdout": stdout,
                "stderr": stderr,
            }

    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return {
            "success": False,
            "timed_out": True,
            "stdout": stdout or "",
            "stderr": f"TIMEOUT after {timeout}s\n" + (stderr or ""),
        }
    except Exception as e:
        return {
            "success": False,
            "timed_out": False,
            "stdout": "",
            "stderr": str(e),
        }


def tail(text: str, n: int = 30) -> str:
    """Return last n lines of a string."""
    lines = (text or "").strip().splitlines()
    return "\n".join(lines[-n:])
