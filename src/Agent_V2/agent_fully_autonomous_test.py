"""
agent_fully_autonomous.py — Fully autonomous research agent.
This version asks the LLM to generate the entire experiment script for each iteration.
Enhanced with: forced arch sequencing, dry-run injection, METRICS guarantee.
"""

import argparse
import os
import re
import pandas as pd
import ast

from llm import OllamaClient
from memory import ExperimentMemory, REQUIRED_ARCHITECTURES
from sandbox import run_experiment, tail
from prompts import DATA_CONTEXT_TEMPLATE, ANALYSIS_PROMPT_TEMPLATE
from templates import (
    SHARED_CONSTRAINTS,
    SHARED_REPAIR_CONSTRAINTS,
    ARCH_TEMPLATES,
    fill_template,
)
import experiment_bow_1 as exp_bow_1
import experiment_bow_2 as exp_bow_2
import experiment_bow_3 as exp_bow_3
import experiment_cnn_1 as exp_cnn_1
import experiment_cnn_2 as exp_cnn_2
import experiment_cnn_3 as exp_cnn_3
import experiment_cnn_4 as exp_cnn_4
import experiment_cnn_5 as exp_cnn_5
import experiment_lstm_1 as exp_lstm_1
import experiment_lstm_2 as exp_lstm_2
import experiment_lstm_3 as exp_lstm_3
import experiment_lstm_4 as exp_lstm_4
import experiment_lstm_5 as exp_lstm_5
import experiment_transformer_1 as exp_transformer_1
import experiment_transformer_2 as exp_transformer_2
import experiment_transformer_3 as exp_transformer_3
import experiment_transformer_4 as exp_transformer_4
import experiment_transformer_5 as exp_transformer_5


# EXPERIMENT
# Set this to run exactly one experiment module file, for example:
# EXPERIMENT = "experiment_lstm_4" or "experiment_lstm_4.py"
EXPERIMENT = "experiment_bow_1"


def _normalize_experiment_filename(value: str | None) -> str | None:
    if not value:
        return None
    name = os.path.basename(value.strip())
    if not name:
        return None
    if not name.endswith(".py"):
        name += ".py"
    return name.lower()


# Architecture-specific hook modules.
EXPERIMENT_HOOKS = {
    "BoW": [exp_bow_1],
    "BoW_advanced": [exp_bow_2],
    "BoW_advanced_thr": [exp_bow_2, exp_bow_3],
    "CNN": [exp_cnn_2, exp_cnn_3, exp_cnn_4, exp_cnn_5, exp_cnn_1],
    "LSTM": [exp_lstm_2, exp_lstm_3, exp_lstm_4, exp_lstm_5, exp_lstm_1],
    "Transformer": [exp_transformer_2, exp_transformer_3, exp_transformer_4, exp_transformer_5, exp_transformer_1],
}

# Selectable test slots: only edit names here.
TEST_EXPERIMENT_NAMES = [
    "BoW_1", "BoW_2", "BoW_3",
    "CNN_1", "CNN_2", "CNN_3", "CNN_4", "CNN_5",
    "LSTM_1", "LSTM_2", "LSTM_3", "LSTM_4", "LSTM_5",
    "Transformer_1", "Transformer_2", "Transformer_3", "Transformer_4", "Transformer_5",
]


def _build_test_experiments(names: list[str]) -> dict[str, tuple[str, list[object]]]:
    module_lookup = {
        "BoW": {1: exp_bow_1, 2: exp_bow_2, 3: exp_bow_3},
        "CNN": {1: exp_cnn_1, 2: exp_cnn_2, 3: exp_cnn_3, 4: exp_cnn_4, 5: exp_cnn_5},
        "LSTM": {1: exp_lstm_1, 2: exp_lstm_2, 3: exp_lstm_3, 4: exp_lstm_4, 5: exp_lstm_5},
        "Transformer": {
            1: exp_transformer_1,
            2: exp_transformer_2,
            3: exp_transformer_3,
            4: exp_transformer_4,
            5: exp_transformer_5,
        },
    }

    arch_override = {
        "BoW_1": "BoW",
        "BoW_2": "BoW_advanced",
        "BoW_3": "BoW_advanced_thr",
    }

    built: dict[str, tuple[str, list[object]]] = {}
    for name in names:
        base, _, idx_str = name.partition("_")
        if not idx_str.isdigit():
            raise ValueError(f"Invalid test experiment name: {name}. Expected format like CNN_3")

        idx = int(idx_str)
        if base not in module_lookup or idx not in module_lookup[base]:
            raise ValueError(f"No module found for test experiment: {name}")

        arch = arch_override.get(name, base)
        if name == "BoW_3":
            built[name] = (arch, [exp_bow_2, exp_bow_3])
        else:
            built[name] = (arch, [module_lookup[base][idx]])

    return built


TEST_EXPERIMENTS = _build_test_experiments(TEST_EXPERIMENT_NAMES)

EXPERIMENT_FILE_TO_MODULE = {
    "experiment_bow_1.py": exp_bow_1,
    "experiment_bow_2.py": exp_bow_2,
    "experiment_bow_3.py": exp_bow_3,
    "experiment_cnn_1.py": exp_cnn_1,
    "experiment_cnn_2.py": exp_cnn_2,
    "experiment_cnn_3.py": exp_cnn_3,
    "experiment_cnn_4.py": exp_cnn_4,
    "experiment_cnn_5.py": exp_cnn_5,
    "experiment_lstm_1.py": exp_lstm_1,
    "experiment_lstm_2.py": exp_lstm_2,
    "experiment_lstm_3.py": exp_lstm_3,
    "experiment_lstm_4.py": exp_lstm_4,
    "experiment_lstm_5.py": exp_lstm_5,
    "experiment_transformer_1.py": exp_transformer_1,
    "experiment_transformer_2.py": exp_transformer_2,
    "experiment_transformer_3.py": exp_transformer_3,
    "experiment_transformer_4.py": exp_transformer_4,
    "experiment_transformer_5.py": exp_transformer_5,
}


def _arch_from_experiment_file(experiment_file: str) -> str:
    file_name = os.path.basename(experiment_file).lower()
    if "transformer" in file_name:
        return "Transformer"
    if "lstm" in file_name:
        return "LSTM"
    if "cnn" in file_name:
        return "CNN"
    if file_name == "experiment_bow_2.py":
        return "BoW_advanced"
    if file_name == "experiment_bow_3.py":
        return "BoW_advanced_thr"
    return "BoW"


MAX_ITERATIONS = 2
TARGET_F1 = 0.88
PLATEAU_WINDOW = 5
MIN_IMPROVEMENT = 0.002
MAX_REPAIR_ATTEMPTS = 2
DATA_DIR_ENV = "DISASTER_AGENT_DATA_DIR"
DEFAULT_DATA_DIR = "data"
GENERATED_CODE_DIR = "generated_code"

# Fully-autonomous exploration order (includes the tuned BoW_advanced variant as a separate step).
ARCH_SEQUENCE = ["BoW", "BoW_advanced", "BoW_advanced_thr", "CNN", "LSTM", "Transformer"]

FULL_SYSTEM = (
    "You are an expert ML engineer writing Python scripts for the Kaggle Disaster Tweets competition.\n"
    "Respond with ONLY a ```python code block. No text outside the code block.\n\n"
    + SHARED_CONSTRAINTS
)
REPAIR_SYSTEM = (
    "You are an expert Python code repair assistant for ML scripts.\n"
    "Respond with ONLY a ```python code block. No text outside the code block.\n\n"
    + SHARED_REPAIR_CONSTRAINTS
)


def get_data_paths() -> tuple[str, str]:
    """Resolve train/test CSV paths from env-configured data directory, with root fallback."""
    data_dir = os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR)
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return train_path, test_path

    # Backward-compatibility for older setups that kept files at repo root.
    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        print("[Data] WARNING: using train.csv/test.csv from repo root; prefer data/train.csv and data/test.csv")
        return "train.csv", "test.csv"

    raise FileNotFoundError(
        "Could not find dataset files. Expected either "
        f"'{train_path}' and '{test_path}', or repo-root train.csv/test.csv."
    )


def build_data_context() -> str:
    train_path, test_path = get_data_paths()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    vc = train["target"].value_counts()
    total = len(train)
    return DATA_CONTEXT_TEMPLATE.format(
        train_rows=len(train),
        test_rows=len(test),
        class_0=vc.get(0, 0),
        class_1=vc.get(1, 0),
        pct_0=100 * vc.get(0, 0) / total,
        pct_1=100 * vc.get(1, 0) / total,
        missing_kw=100 * train["keyword"].isna().mean(),
        missing_loc=100 * train["location"].isna().mean(),
    )


def write_generated_code(name: str, code: str, stage: str = "latest") -> str:
    """Persist generated script code to a Python file for inspection/debugging."""
    os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
    file_name = f"{name}_{stage}.py"
    file_path = os.path.join(GENERATED_CODE_DIR, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    return file_path


def infer_architecture(text: str) -> str:
    text_lower = text.lower()
    if "bow_advanced_thr" in text_lower or "bowadvanced2" in text_lower or "threshold tuning" in text_lower:
        return "BoW_advanced_thr"
    if any(w in text_lower for w in ["transformer", "bert", "distilbert", "deberta"]):
        return "Transformer"
    if any(w in text_lower for w in ["lstm", "gru", "bilstm", "birnn"]):
        return "LSTM"
    if any(w in text_lower for w in ["conv1d", "cnn", "convolutional"]):
        return "CNN"
    if "ensemble" in text_lower or "bow_advanced" in text_lower:
        return "BoW_advanced"
    return "BoW"


def syntax_check(code: str) -> tuple[bool, str]:
    """Check Python syntax before running. Returns (ok, error_message)."""
    import ast
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


def _preflight_for_arch(code: str, arch: str) -> list[str]:
    issues: list[str] = []
    for module in EXPERIMENT_HOOKS.get(arch, []):
        issues.extend(module.preflight_issues(code, arch))
    return issues


def _autofix_for_arch(code: str, arch: str) -> str:
    fixed_code = code
    for module in EXPERIMENT_HOOKS.get(arch, []):
        fixed_code = module.apply_light_autofixes(fixed_code, arch)
    return fixed_code


def _repair_hint_for_arch(arch: str, stderr_text: str) -> str:
    hints: list[str] = []
    for module in EXPERIMENT_HOOKS.get(arch, []):
        hint = module.build_repair_hint(stderr_text)
        if hint:
            hints.append(hint)
    return "".join(hints)


def _primary_experiment_module(arch: str):
    modules = EXPERIMENT_HOOKS.get(arch, [])
    return modules[-1] if modules else None


def _arch_prompt_for_arch(arch: str) -> str:
    modules = EXPERIMENT_HOOKS.get(arch, [])
    for module in reversed(modules):
        if hasattr(module, "get_arch_prompt"):
            prompt = module.get_arch_prompt()
            if prompt:
                return prompt
    return "Propose an improved version of the best model so far."


def _model_prompt_for_arch(arch: str) -> str:
    modules = EXPERIMENT_HOOKS.get(arch, [])
    for module in reversed(modules):
        if hasattr(module, "get_model_prompt"):
            prompt = module.get_model_prompt()
            if prompt:
                return prompt
    return ""


def _template_for_arch(arch: str) -> str | None:
    modules = EXPERIMENT_HOOKS.get(arch, [])
    for module in reversed(modules):
        if hasattr(module, "get_template"):
            template = module.get_template(ARCH_TEMPLATES)
            if template:
                return template
    return None


def preflight_issues(code: str, arch: str) -> list[str]:
    """Detect common autonomous-generation failures before execution."""
    issues: list[str] = []

    # Reject prompt/requirements leakage into the code body.
    if re.search(r"(?m)^(MANDATORY_SCRIPT_REQUIREMENTS|MANDATORY SCRIPT REQUIREMENTS|DATA LOADING|CROSS-VALIDATION|SUBMISSION|METRICS)\s*:\s*$", code):
        issues.append("Prompt leakage: remove headings like 'DATA LOADING:' / 'CROSS-VALIDATION:' / 'SUBMISSION:' / 'METRICS:' from the script (code only)")

    # Common failure: wrapping runtime flags into a dict and then referencing undefined DRY_RUN/N_SPLITS.
    if re.search(r"(?m)^\s*RUNTIME_FLAGS\s*=\s*{", code):
        issues.append("Forbidden runtime flags: define DRY_RUN and N_SPLITS as plain variables (no RUNTIME_FLAGS dict)")

    if "hstack(" in code and "from scipy.sparse import hstack" not in code:
        issues.append("Missing import: from scipy.sparse import hstack")

    if re.search(r"\brankdata\(", code) and "from scipy.stats import rankdata" not in code:
        issues.append("Missing import: from scipy.stats import rankdata")

    issues.extend(_preflight_for_arch(code, arch))

    return issues


def apply_light_autofixes(code: str, arch: str) -> str:
    return _autofix_for_arch(code, arch)


def get_not_yet_successful(memory: ExperimentMemory) -> list[str]:
    successful_arches = {e["architecture"] for e in memory.experiments if e.get("success")}
    return [a for a in ARCH_SEQUENCE if a not in successful_arches]


def get_not_yet_attempted(memory: ExperimentMemory) -> list[str]:
    tried_arches = set(memory.get_tried_architectures())
    return [a for a in ARCH_SEQUENCE if a not in tried_arches]


def request_repair(
    llm: OllamaClient,
    arch: str,
    name: str,
    failed_code: str,
    stderr_text: str,
    stdout_text: str,
    attempt: int,
) -> str:
    extra_hint = _repair_hint_for_arch(arch, stderr_text)

    repair_prompt = (
        f"Repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS} for architecture: {arch}.\n"
        f"Keep submission path exactly: submissions/{name}_submission.csv\n\n"
        "Execution failed. Fix the script and return a fully executable Python script only.\n"
        "Do not include requirement prose, bullet lists, or copied prompt text inside the code block.\n\n"
        f"STDERR:\n{stderr_text}\n\n"
        f"{extra_hint}"
        f"STDOUT (tail):\n{tail(stdout_text, 30)}\n\n"
        "FAILED CODE:\n"
        "```python\n"
        f"{failed_code}\n"
        "```\n"
    )
    _, fixed_code = llm.propose(REPAIR_SYSTEM, repair_prompt)
    return fixed_code


def main(
    model: str,
    max_iterations: int,
    persist: bool = True,
    selected_experiment: str | None = None,
    selected_test_experiment: str | None = None,
    selected_experiment_file: str | None = None,
):
    print("\n" + "=" * 60)
    print("  FULLY AUTONOMOUS ML RESEARCH AGENT - Disaster Tweets")
    print("=" * 60)

    os.makedirs("submissions", exist_ok=True)
    data_context = build_data_context()
    memory = ExperimentMemory(persist=persist)
    llm = OllamaClient(model=model)

    effective_experiment_file = _normalize_experiment_filename(selected_experiment_file) or _normalize_experiment_filename(EXPERIMENT)

    run_iterations = 1 if (selected_experiment or selected_test_experiment or effective_experiment_file) else max_iterations
    print(f"\n[Agent] Starting loop. Max iterations: {run_iterations}")
    print(f"[Agent] Target F1: {TARGET_F1}  |  Plateau window: {PLATEAU_WINDOW}\n")
    if selected_experiment:
        print(f"[Agent] Single experiment mode enabled: {selected_experiment}")
    if selected_test_experiment:
        print(f"[Agent] Single test slot mode enabled: {selected_test_experiment}")
    if effective_experiment_file:
        print(f"[Agent] Single experiment file mode enabled: {effective_experiment_file}")

    for iteration in range(1, run_iterations + 1):
        print(f"\n{'-'*60}")
        print(f"  ITERATION {iteration}/{run_iterations}")
        print(f"{'-'*60}")

        tried = memory.get_tried_architectures()
        # Follow the fixed architecture sequence, without repeating already-attempted architectures.
        not_yet = get_not_yet_attempted(memory)
        best = memory.best()
        best_f1 = best["metrics"].get("f1", 0) if best else 0

        print(f"[Agent] Tried: {tried or 'none'}  |  Still needed (sequence): {not_yet or 'all covered'}")
        print(f"[Agent] Best F1: {best_f1:.5f}" if best_f1 else "[Agent] No successful runs yet")

        if effective_experiment_file:
            if effective_experiment_file not in EXPERIMENT_FILE_TO_MODULE:
                raise ValueError(
                    "Unknown experiment file: "
                    f"{effective_experiment_file}. Use one of: {', '.join(sorted(EXPERIMENT_FILE_TO_MODULE.keys()))}"
                )
            next_arch = _arch_from_experiment_file(effective_experiment_file)
            EXPERIMENT_HOOKS[next_arch] = [EXPERIMENT_FILE_TO_MODULE[effective_experiment_file]]
        elif selected_test_experiment:
            next_arch, selected_modules = TEST_EXPERIMENTS[selected_test_experiment]
            EXPERIMENT_HOOKS[next_arch] = selected_modules
        else:
            next_arch = selected_experiment if selected_experiment else (not_yet[0] if not_yet else "BoW_advanced_thr")
        name = (
            f"manual_{os.path.splitext(os.path.basename(effective_experiment_file))[0].lower()}"
            if effective_experiment_file
            else (
                f"manual_{selected_test_experiment.lower()}"
                if selected_test_experiment
                else (
                    f"manual_{next_arch.lower().replace('_', '')}"
                    if selected_experiment
                    else (
                        f"iter{iteration:02d}_bowadvanced2"
                        if next_arch == "BoW_advanced_thr"
                        else f"iter{iteration:02d}_{next_arch.lower().replace('_', '')}"
                    )
                )
            )
        )

        template = _template_for_arch(next_arch)
        if template:
            # Template-based: avoid fragile full-script generation for BoW variants.
            print(f"\n[THINK] Asking LLM to write model function for: {next_arch}...")
            model_prompt = _model_prompt_for_arch(next_arch)
            response, model_code = llm.propose(
                "You are a Python coding assistant. Write ONLY the requested function, nothing else.",
                model_prompt,
            )
            model_code = model_code or ""

            def _accept_function_only(code_text: str, fn_name: str, arg_count: int | None) -> bool:
                """
                Only accept code that defines exactly one top-level function.
                Prevents full-script injection into templates.
                """
                try:
                    tree = ast.parse(code_text)
                except SyntaxError:
                    return False

                body = list(tree.body)
                # Allow (optional) module docstring + one function def.
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                    body = body[1:]

                if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
                    return False
                if body[0].name != fn_name:
                    return False
                if arg_count is not None and len(body[0].args.args) != arg_count:
                    return False
                return True

            if model_code:
                if next_arch in ("BoW_advanced", "BoW_advanced_thr"):
                    if not _accept_function_only(model_code, "get_ensemble_probs", 4):
                        model_code = ""
                elif next_arch == "BoW":
                    if not _accept_function_only(model_code, "build_model", None):
                        model_code = ""
                elif next_arch in ("CNN", "LSTM"):
                    if not _accept_function_only(model_code, "build_model", 1):
                        model_code = ""

            code = fill_template(template, model_code, name)
            user_prompt = model_prompt
        else:
            print(f"\n[THINK] Asking LLM to write: {next_arch}...")
            arch_instruction = _arch_prompt_for_arch(next_arch)
            user_prompt = (
                f"Write a complete Python script implementing this architecture: {next_arch}\n\n"
                f"ARCHITECTURE DETAILS: {arch_instruction}\n\n"
                f"{data_context}\n"
                f"Experiment name for submission file: {name}\n"
                f"Submission path: submissions/{name}_submission.csv\n\n"
                f"Past experiments (do not repeat failures):\n{memory.get_history_summary()}\n"
            )
            response, code = llm.propose(FULL_SYSTEM, user_prompt)

        if not code:
            print("[THINK] LLM returned no code. Skipping iteration.")
            memory.add(
                name=f"iter{iteration:02d}_no_code", architecture="unknown",
                prompt_sent=user_prompt, code="", stdout="",
                stderr="LLM returned no code block", metrics={},
                analysis="LLM failed to produce a code block.", success=False,
            )
            continue

        initial_code_path = write_generated_code(name, code, stage="initial")
        print(f"[THINK] Generated code saved: {initial_code_path}")

        arch = next_arch if template else infer_architecture(response + next_arch)
        print(f"[THINK] Architecture: {arch}  |  Name: {name}")

        attempt = 0
        result = None
        run_code = code
        repaired = False
        while attempt <= MAX_REPAIR_ATTEMPTS:
            run_code = apply_light_autofixes(run_code, arch)
            issues = preflight_issues(run_code, arch)
            if issues:
                if attempt == MAX_REPAIR_ATTEMPTS:
                    result = {
                        "success": False,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": "Preflight validation failed:\n- " + "\n- ".join(issues),
                        "metrics": {},
                    }
                    break
                print(f"[REPAIR] Preflight issue detected. Asking LLM to repair (attempt {attempt + 1})...")
                fixed = request_repair(
                    llm=llm,
                    arch=arch,
                    name=name,
                    failed_code=run_code,
                    stderr_text="Preflight validation failed:\n- " + "\n- ".join(issues),
                    stdout_text="",
                    attempt=attempt + 1,
                )
                if not fixed:
                    result = {
                        "success": False,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": "Preflight validation failed:\n- " + "\n- ".join(issues),
                        "metrics": {},
                    }
                    break
                run_code = fixed
                repaired_path = write_generated_code(name, run_code, stage=f"repair_{attempt + 1}")
                print(f"[REPAIR] Updated code saved: {repaired_path}")
                repaired = True
                attempt += 1
                continue

            ok, syntax_err = syntax_check(run_code)
            if not ok:
                if attempt == MAX_REPAIR_ATTEMPTS:
                    result = {
                        "success": False,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": f"SyntaxError: {syntax_err}",
                        "metrics": {},
                    }
                    break
                print(f"[REPAIR] Syntax issue detected. Asking LLM to repair (attempt {attempt + 1})...")
                fixed = request_repair(
                    llm=llm,
                    arch=arch,
                    name=name,
                    failed_code=run_code,
                    stderr_text=f"SyntaxError: {syntax_err}",
                    stdout_text="",
                    attempt=attempt + 1,
                )
                if not fixed:
                    result = {
                        "success": False,
                        "timed_out": False,
                        "stdout": "",
                        "stderr": f"SyntaxError: {syntax_err}",
                        "metrics": {},
                    }
                    break
                run_code = fixed
                repaired_path = write_generated_code(name, run_code, stage=f"repair_{attempt + 1}")
                print(f"[REPAIR] Updated code saved: {repaired_path}")
                repaired = True
                attempt += 1
                continue

            print("\n[EXECUTE] Running experiment...")
            result = run_experiment(run_code, name)
            if result["success"]:
                if repaired:
                    print(f"[REPAIR] Successfully repaired on attempt {attempt}.")
                break

            if attempt == MAX_REPAIR_ATTEMPTS:
                break

            print(f"[REPAIR] Run failed. Asking LLM to repair (attempt {attempt + 1})...")
            fixed = request_repair(
                llm=llm,
                arch=arch,
                name=name,
                failed_code=run_code,
                stderr_text=result["stderr"],
                stdout_text=result["stdout"],
                attempt=attempt + 1,
            )
            if not fixed:
                break
            run_code = fixed
            repaired_path = write_generated_code(name, run_code, stage=f"repair_{attempt + 1}")
            print(f"[REPAIR] Updated code saved: {repaired_path}")
            repaired = True
            attempt += 1

        code = run_code
        final_code_path = write_generated_code(name, code, stage="final")
        print(f"[Agent] Final code saved: {final_code_path}")

        if not result:
            result = {
                "success": False,
                "timed_out": False,
                "stdout": "",
                "stderr": "Execution failed and repair returned no code.",
                "metrics": {},
            }

        if not result["success"]:
            print("\n--- FULL CODE ---")
            print(code)
            print("\n--- STDERR ---")
            print(result.get("stderr", ""))
            print("\n--- STDOUT ---")
            print(result.get("stdout", ""))

        # ── REFLECT ──────────────────────────────────────────────
        print("\n[REFLECT] Asking LLM to analyze results...")
        status = "success" if result["success"] else ("timeout" if result["timed_out"] else "crash")
        analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            name=name,
            status=status,
            metrics=result["metrics"] or "none",
            stdout_tail=tail(result["stdout"], 30),
            stderr_tail=tail(result["stderr"], 20),
        )
        analysis = llm.analyze(analysis_prompt)

        # ── Update memory ─────────────────────────────────────────
        memory.add(
            name=name, architecture=arch,
            prompt_sent=user_prompt, code=code,
            stdout=result["stdout"], stderr=result["stderr"],
            metrics=result["metrics"], analysis=analysis,
            success=result["success"],
        )

        f1 = result["metrics"].get("f1", None)
        print(f"\n[Result] {name} | F1={f1:.5f}" if f1 else f"\n[Result] {name} | FAILED ({status})")
        print(f"[Analysis] {analysis[:200].strip()}")

        if f1 and f1 >= TARGET_F1:
            print(f"\n[Agent] TARGET F1 {TARGET_F1} REACHED! Stopping early.")
            break
        if memory.is_plateau(PLATEAU_WINDOW, MIN_IMPROVEMENT):
            print(f"\n[Agent] Plateau detected. Stopping.")
            break

    memory.print_leaderboard()
    best = memory.best()
    if best:
        print(f"\n[Agent] Best submission: submissions/{best['name']}_submission.csv")
    print("\n[Agent] Done. Full log saved to experiment_log.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Autonomous ML Research Agent")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:3b")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS)
    parser.add_argument(
        "--experiment",
        type=str,
        choices=ARCH_SEQUENCE,
        help="Run only one selected experiment architecture.",
    )
    parser.add_argument(
        "--test-experiment",
        type=str,
        choices=sorted(TEST_EXPERIMENTS.keys()),
        help="Run only one selected test slot (e.g., BoW_1, CNN_3, LSTM_5).",
    )
    parser.add_argument(
        "--experiment-file",
        type=str,
        help="Run only one specific experiment module file (e.g., experiment_cnn_1.py). If omitted, uses EXPERIMENT constant when set.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore and do not write experiment_log.json (start sequence from scratch).",
    )
    args = parser.parse_args()
    if args.fresh:
        print("\n[Agent] Running in --fresh mode (no experiment_log.json read/write).")
    configured_experiment_file = _normalize_experiment_filename(EXPERIMENT)
    selected_modes = [bool(args.experiment), bool(args.test_experiment), bool(args.experiment_file or configured_experiment_file)]
    if sum(selected_modes) > 1:
        parser.error("Use only one of --experiment, --test-experiment, or --experiment-file.")
    main(
        model=args.model,
        max_iterations=args.max_iter,
        persist=not args.fresh,
        selected_experiment=args.experiment,
        selected_test_experiment=args.test_experiment,
        selected_experiment_file=args.experiment_file or configured_experiment_file,
    )
