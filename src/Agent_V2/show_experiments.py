import json

with open("experiment_log.json", encoding="utf-8") as f:
    exps = json.load(f)

for e in exps:
    status = "SUCCESS" if e["success"] else "FAILED"
    f1 = e["metrics"].get("f1", "N/A")
    print(f"\n{'='*60}")
    print(f"[{e['id']}] {e['name']} | {status} | F1={f1}")
    print(f"Architecture: {e['architecture']}")
    print(f"Timestamp: {e['timestamp']}")
    if e["success"]:
        print(f"\n--- GENERATED CODE ---")
        print(e["code_generated"])
    else:
        print(f"\n--- ERROR ---")
        print(e["stderr"][:500])
    if e.get("llm_analysis"):
        print(f"\n--- LLM ANALYSIS ---")
        print(e["llm_analysis"][:400])
