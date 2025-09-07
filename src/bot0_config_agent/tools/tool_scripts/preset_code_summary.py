"""
bot0_config_agent/tools/tool_scripts/preset_code_summary.py

Create an LLM-ready prompt for summarizing code. Consumes file contents (from read_files)
and emits a single prompt string tailored for codebases (classes, functions, imports, etc.).
"""

# Standard & 3rd party
from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
from textwrap import shorten

# From project modules
from bot0_config_agent.tools.configs.tool_models import FileContent
from bot0_config_agent.agent_models.step_status import StepStatus


DEFAULT_HEADER = """You are a senior code analyst. Summarize the following codebase.
Focus on purpose, key modules, public interfaces, data models, important classes/functions, and how pieces fit.
Call out external deps (imports), configuration, and potential risks (I/O, security, concurrency).
Keep it concise and technically precise."""


# Simple, language-agnostic “signal” extractors (fast + robust)
def _first_lines(code: str, n: int = 20) -> str:
    return "\n".join(code.splitlines()[:n]).strip()


def _count_occurrences(code: str, needles: List[str]) -> Dict[str, int]:
    low = code.lower()
    return {k: low.count(k) for k in needles}


def _file_sketch(entry: FileContent, max_chars_per_file: int) -> str:
    code = entry.content or ""
    head = _first_lines(code, 30)
    stats = _count_occurrences(
        code,
        needles=[
            "class ",
            "def ",
            "function ",
            "interface ",
            "struct ",
            "import ",
            "from ",
            "package ",
            "module ",
        ],
    )
    # Trim for budget
    trimmed = shorten(code, width=max_chars_per_file, placeholder="\n…(truncated)…")
    # Compact, LLM-friendly section
    return (
        f"##### FILE: {entry.file}\n"
        f"// quick-signals: {stats}\n"
        f"// header preview:\n{head}\n\n"
        f"{trimmed}\n"
    )


def _compose_prompt(
    files: List[FileContent],
    *,
    task: Literal["summarize", "review", "document"] = "summarize",
    style: Optional[str] = None,
    max_chars: int = 8000,
) -> str:
    header = DEFAULT_HEADER
    if task == "review":
        header = (
            header.replace("Summarize", "Review")
            + "\nAdd brief improvement suggestions."
        )
    elif task == "document":
        header = (
            header.replace("Summarize", "Document")
            + "\nEmit a short, clean README-style outline."
        )

    if style:
        header += f"\nStyle: {style.strip()}"

    # Fair-share budget per file (leave ~1/6 for header/joiners)
    per_file = max(500, int(max_chars * 5 / 6 / max(1, len(files))))

    body_chunks = [_file_sketch(f, per_file) for f in files]
    prompt = f"{header}\n\n" + "\n".join(body_chunks)
    # Final clamp
    if len(prompt) > max_chars:
        prompt = shorten(prompt, width=max_chars, placeholder="\n…(end truncated)…")
    return prompt


# === PUBLIC TOOL ENTRY ===
def preset_code_summary(
    *,
    files: List[Dict[str, str]] | List[FileContent],
    task: str = "summarize",
    max_chars: Optional[int] = 8000,
    style: Optional[str] = None,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Params:
      - files: list of {"file": str, "content": str} (typically from read_files)
      - task:  "summarize" | "review" | "document"
      - max_chars: soft cap for prompt length
      - style: optional tone or format guidance
      - include_metadata: include tiny per-file 'quick-signals' header

    Returns (standard envelope):
      {
        "status": StepStatus.SUCCESS | StepStatus.ERROR,
        "message": str,
        "result": {
          "prompt": str,
          "meta": { "file_count": int, "total_chars": int }
        }
      }
    """
    try:
        # Normalize into FileContent models (duck-type compatible)
        fc_list: List[FileContent] = []
        for item in files:
            if isinstance(item, FileContent):
                fc_list.append(item)
            else:
                # tolerate {'path':..., 'text':...} variants
                file = item.get("file") or item.get("path") or ""
                content = item.get("content") or item.get("text") or ""
                fc_list.append(FileContent(file=file, content=content))

        if not fc_list:
            return {
                "status": StepStatus.ERROR,
                "message": "No files provided to preset_code_summary.",
                "result": None,
            }

        # Compose specialized code prompt
        prompt = _compose_prompt(
            fc_list,
            task=task if task in ("summarize", "review", "document") else "summarize",
            style=style,
            max_chars=int(max_chars or 8000),
        )

        meta = {
            "file_count": len(fc_list),
            "total_chars": sum(len(f.content or "") for f in fc_list),
        }

        # result.prompt is intentionally top-level inside result so transforms can
        # pass it through
        return {
            "status": StepStatus.SUCCESS,
            "message": f"Prepared code {task} prompt for {len(fc_list)} file(s).",
            "result": (
                {"prompt": prompt, "meta": meta}
                if include_metadata
                else {"prompt": prompt}
            ),
        }

    except Exception as e:
        return {
            "status": StepStatus.ERROR,
            "message": f"preset_code_summary failed: {e}",
            "result": None,
        }
