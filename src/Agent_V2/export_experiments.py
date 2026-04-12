"""
export_experiments.py
Saves each experiment's generated code to a .py file in experiments/ folder.
Also prints a summary table.
"""
import json
import os

with open("experiment_log.json", encoding="utf-8") as f:
    exps = json.load(f)

os.makedirs("experiments", exist_ok=True)

print(f"{'ID':<4} {'Name':<30} {'Arch':<14} {'F1':<8} {'Status'}")
print("-" * 70)

for e in exps:
    f1 = e["metrics"].get("f1", "N/A") if e["metrics"] else "N/A"
    status = "OK" if e["success"] else "FAILED"
    print(f"{e['id']:<4} {e['name'][:29]:<30} {e['architecture']:<14} {str(f1):<8} {status}")

    if e["code_generated"]:
        path = f"experiments/{e['id']:02d}_{e['name']}.py"
        with open(path, "w", encoding="utf-8") as f_out:
            f_out.write(f"# Experiment: {e['name']}\n")
            f_out.write(f"# Architecture: {e['architecture']}\n")
            f_out.write(f"# F1: {f1} | Status: {status}\n")
            f_out.write(f"# Timestamp: {e['timestamp']}\n")
            f_out.write(f"# LLM Analysis: {(e['llm_analysis'] or '')[:200]}\n\n")
            f_out.write(e["code_generated"])
        print(f"       -> saved to {path}")

print(f"\nTotal: {len(exps)} experiments")
