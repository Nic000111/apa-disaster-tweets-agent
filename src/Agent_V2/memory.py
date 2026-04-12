"""
memory.py — Tiered experiment memory (matches the diagram).

Tier 1 — Project brief: static context (set once, always included)
Tier 2 — Rolling log: last 15 experiment summaries (prevents repeats)
Milestones — Best results kept forever (never evicted)

Backed by experiment_log.json so the agent can resume across restarts.
"""

import json
import os
from datetime import datetime

LOG_PATH = "experiment_log.json"
ROLLING_WINDOW = 15  # how many recent experiments to include in prompts

# Architecture families the agent must cover (in order).
# BoW_advanced_thr is a follow-up to BoW_advanced that tunes the decision threshold on OOF probs.
REQUIRED_ARCHITECTURES = ["BoW", "BoW_advanced", "BoW_advanced_thr", "CNN", "LSTM", "Transformer"]


class ExperimentMemory:
    def __init__(self, persist: bool = True, log_path: str = LOG_PATH):
        self.persist = persist
        self.log_path = log_path
        self.experiments = []
        self._load()

    def _load(self):
        if not self.persist:
            self.experiments = []
            return
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                self.experiments = json.load(f)
            print(f"[Memory] Loaded {len(self.experiments)} past experiments from {self.log_path}")
        else:
            self.experiments = []

    def _save(self):
        if not self.persist:
            return
        with open(self.log_path, "w") as f:
            json.dump(self.experiments, f, indent=2)

    def add(self, name: str, architecture: str, prompt_sent: str,
            code: str, stdout: str, stderr: str,
            metrics: dict, analysis: str, success: bool):
        record = {
            "id": len(self.experiments) + 1,
            "name": name,
            "architecture": architecture,
            "prompt_sent": prompt_sent,
            "code_generated": code,
            "stdout": stdout,
            "stderr": stderr,
            "metrics": metrics,
            "llm_analysis": analysis,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
        self.experiments.append(record)
        self._save()

    def get_history_summary(self) -> str:
        """
        Compact summary of last ROLLING_WINDOW experiments.
        This goes into every proposal prompt so the LLM never repeats failures.
        """
        if not self.experiments:
            return "No experiments run yet. Start fresh!"

        recent = self.experiments[-ROLLING_WINDOW:]
        lines = []
        for e in recent:
            f1 = e["metrics"].get("f1", "N/A") if e["metrics"] else "N/A"
            status = "OK" if e["success"] else "FAILED"
            analysis_short = (e["llm_analysis"] or "")[:120].replace("\n", " ")
            lines.append(
                f"  [{e['id']}] {e['name']} ({e['architecture']}) | F1={f1} | {status} | {analysis_short}"
            )

        # Always include the milestone (best result ever)
        best = self.best()
        if best:
            best_f1 = best["metrics"].get("f1", "N/A")
            lines.append(f"\n  *** BEST SO FAR: {best['name']} | F1={best_f1} ***")

        return "\n".join(lines)

    def get_tried_architectures(self) -> list[str]:
        return list({e["architecture"] for e in self.experiments})

    def get_not_yet_tried(self) -> list[str]:
        tried = self.get_tried_architectures()
        return [a for a in REQUIRED_ARCHITECTURES if a not in tried]

    def has_tried(self, architecture: str) -> bool:
        return architecture in self.get_tried_architectures()

    def best(self) -> dict | None:
        successful = [e for e in self.experiments if e["success"] and e["metrics"].get("f1")]
        if not successful:
            return None
        return max(successful, key=lambda e: e["metrics"].get("f1", 0))

    def count(self) -> int:
        return len(self.experiments)

    def is_plateau(self, window: int = 5, min_improvement: float = 0.002) -> bool:
        """
        Returns True if F1 hasn't improved by at least min_improvement
        in the last `window` experiments. Used as a stopping criterion.
        """
        recent = [
            e for e in self.experiments[-window:]
            if e["success"] and e["metrics"].get("f1")
        ]
        if len(recent) < window:
            return False
        scores = [e["metrics"]["f1"] for e in recent]
        return (max(scores) - min(scores)) < min_improvement

    def print_leaderboard(self):
        successful = [e for e in self.experiments if e["success"] and e["metrics"].get("f1")]
        if not successful:
            print("[Memory] No successful experiments yet.")
            return
        ranked = sorted(successful, key=lambda e: e["metrics"].get("f1", 0), reverse=True)
        print("\n" + "=" * 60)
        print("EXPERIMENT LEADERBOARD")
        print("=" * 60)
        print(f"{'Rank':<5} {'Name':<35} {'Arch':<12} {'F1':<8}")
        print("-" * 60)
        for i, e in enumerate(ranked, 1):
            f1 = e["metrics"].get("f1", 0)
            print(f"{i:<5} {e['name'][:34]:<35} {e['architecture']:<12} {f1:.5f}")
        print("=" * 60)
