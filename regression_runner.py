#!/usr/bin/env python3
"""
regression_runner.py
Runs a set of CLI regression tests and summarizes whether they "worked".

Definition of "working" (tunable below):
- exit code == 0
- AND stdout contains either:
  * 'RESULT_JSON for step' (your agent prints these), OR
  * a JSON-y success flag like `"status": "success"`, OR
  * 'Plan execution complete' (executor banner)
"""

from __future__ import annotations
import subprocess, shlex, time, re, os, sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

# -------- Success heuristics (adjust as needed) ----------
SUCCESS_PATTERNS = [
    re.compile(r'RESULT_JSON for step', re.IGNORECASE),
    re.compile(r'"status"\s*:\s*"success"', re.IGNORECASE),
    re.compile(r'Plan execution complete', re.IGNORECASE),
]

def looks_successful(stdout: str) -> Tuple[bool, str]:
    for pat in SUCCESS_PATTERNS:
        if pat.search(stdout or ""):
            return True, f"matched /{pat.pattern}/"
    return False, "no success patterns matched"

# -------- Test definitions --------------------------------
@dataclass
class TestCase:
    group: str
    name: str
    cmd: str
    timeout: int = 180  # seconds

TESTS: List[TestCase] = [
    # ===== GPT (Working) =====
    TestCase("GPT (API)", "YAML summary", 'python -m agent.cli --api-model gpt-4.1-mini --once "Find all YAML files in the project and then create a summary of their contents."'),
    TestCase("GPT (API)", "List ./agent excl", 'python -m agent.cli --api-model gpt-4.1-mini --once "List all files in the ./agent directory excluding __pycache__, .git, and venv."'),
    TestCase("GPT (API)", "Locate planner.py", 'python -m agent.cli --api-model gpt-4.1-mini --once "is there a file called planner.py in the project"'),
    TestCase("GPT (API)", "Count files in agent", 'python -m agent.cli --api-model gpt-4.1-mini --once "count the files in the agent directory"'),

    # ===== GEMINI (Working) =====
    TestCase("GEMINI (API)", "Count files in agent", 'python -m agent.cli --api-model gemini-1.5-flash-latest --once "count the files in the agent directory"'),
    TestCase("GEMINI (API)", "List ./agent excl", 'python -m agent.cli --api-model gemini-1.5-flash-latest --once "List all files in the ./agent directory excluding __pycache__, .git, and venv."'),

    # ===== LLAMA 2 7B GPTQ (Working) =====
    TestCase("LLAMA_2_7B_GPTQ (local)", "Count files in tools", 'python -m agent.cli --local-model llama_2_7b_chat_gptq --once "count the number of files in the tools folder"'),

    # ===== LLAMA 3.2 3B GPTQ (Working) =====
    TestCase("LLAMA_3_2_3B_GPTQ (local)", "Count files in agent", 'python -m agent.cli --local-model llama_3_2_3b_gptq --once "count the number of files in the agent folder"'),
    TestCase("LLAMA_3_2_3B_GPTQ (local)", "YAML summary", 'python -m agent.cli --local-model llama_3_2_3b_gptq --once "Find all YAML files in the project and then create a summary of their contents."'),

    # ===== LIQUID (NOT WORKING per note) =====
    TestCase("LIQUID lfm2_1_2b (local)", "List ./agent (miscounts ~10)", 'python -m agent.cli --local-model lfm2_1_2b --once "List all files in the ./agent list all files"'),
    TestCase("LIQUID lfm2_1_2b (local)", "Count files in agent", 'python -m agent.cli --local-model lfm2_1_2b --once "count all files in agent folder"'),
    TestCase("LIQUID lfm2_1_2b (local)", "Count files in project", 'python -m agent.cli --local-model lfm2_1_2b --once "count the number of files in this project"'),
    TestCase("LIQUID lfm2_1_2b (local)", "List files in prompts", 'python -m agent.cli --local-model lfm2_1_2b --once "list files in prompts folder"'),
]

# -------- Runner ------------------------------------------
@dataclass
class TestResult:
    group: str
    name: str
    cmd: str
    ok: bool
    reason: str
    rc: int
    secs: float

def run_test(tc: TestCase) -> TestResult:
    t0 = time.time()
    try:
        proc = subprocess.run(
            tc.cmd if os.name == "nt" else shlex.split(tc.cmd),
            capture_output=True, text=True, timeout=tc.timeout
        )
        dt = time.time() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        matched, why = looks_successful(stdout)
        ok = (proc.returncode == 0) and matched
        reason = why if ok else f"rc={proc.returncode}; {why}; stderr_len={len(stderr)}"
        return TestResult(tc.group, tc.name, tc.cmd, ok, reason, proc.returncode, dt)
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        return TestResult(tc.group, tc.name, tc.cmd, False, "timeout", 124, dt)
    except FileNotFoundError as e:
        dt = time.time() - t0
        return TestResult(tc.group, tc.name, tc.cmd, False, f"cmd not found: {e}", 127, dt)
    except Exception as e:
        dt = time.time() - t0
        return TestResult(tc.group, tc.name, tc.cmd, False, f"exception: {e}", 1, dt)

def summarize(results: List[TestResult]) -> str:
    # Grouped markdown table
    lines = ["# Regression Summary", ""]
    by_group: Dict[str, List[TestResult]] = {}
    for r in results:
        by_group.setdefault(r.group, []).append(r)

    totals_ok = 0
    for grp, items in by_group.items():
        ok = sum(1 for r in items if r.ok)
        totals_ok += ok
        lines.append(f"## {grp} — {ok}/{len(items)} passing")
        lines.append("")
        lines.append("| Test | Result | Time (s) | Note |")
        lines.append("|---|---|---:|---|")
        for r in items:
            status = "✅ WORKING" if r.ok else "❌ FAIL"
            lines.append(f"| {r.name} | {status} | {r.secs:.2f} | {r.reason} |")
        lines.append("")

    lines.append(f"**TOTAL:** {totals_ok}/{len(results)} passing")
    lines.append("")
    return "\n".join(lines)

def main() -> int:
    print("Running regression tests...\n")
    results = [run_test(tc) for tc in TESTS]
    md = summarize(results)
    out = "regression_summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\nSaved summary to {out}")
    # exit nonzero if any fail (useful in CI)
    return 0 if all(r.ok for r in results) else 1

if __name__ == "__main__":
    sys.exit(main())

