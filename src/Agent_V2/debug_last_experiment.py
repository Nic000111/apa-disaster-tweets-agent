"""
debug_last_experiment.py
Shows the full details of the most recent failed experiment:
- What code was generated
- What the dry run stderr was
- What the LLM analysis said
"""
import json

with open("experiment_log.json", encoding="utf-8") as f:
    exps = json.load(f)

# Get last failed experiment
failed = [e for e in exps if not e["success"]]
if not failed:
    print("No failed experiments found.")
    exit()

e = failed[-1]

print("=" * 60)
print(f"LAST FAILED EXPERIMENT: {e['name']}")
print(f"Architecture: {e['architecture']}")
print(f"Timestamp: {e['timestamp']}")
print("=" * 60)

print("\n--- GENERATED CODE ---")
print(e["code_generated"] if e["code_generated"] else "(no code was generated)")

print("\n--- STDOUT ---")
print(e["stdout"] if e["stdout"] else "(empty)")

print("\n--- STDERR ---")
print(e["stderr"] if e["stderr"] else "(empty)")

print("\n--- LLM ANALYSIS ---")
print(e["llm_analysis"] if e["llm_analysis"] else "(none)")

print("\n--- PROMPT SENT TO LLM ---")
print(e["prompt_sent"][:1000] + "..." if len(e["prompt_sent"]) > 1000 else e["prompt_sent"])
