# tools/aggregate_file_content.py
"""
Aggregates file contents from prior read_files results (MCP-compatible).
Handles dict, JSON-string, and literal-eval step formats robustly.
Limits to 20 files and 1MB total size.
"""

from __future__ import annotations
import json
import ast
import logging
from typing import Any, Dict, List
from agent_models.step_status import StepStatus

logger = logging.getLogger(__name__)

MAX_FILES = 20
MAX_TOTAL_BYTES = 1_000_000  # 1 MB total cap


def _json_safe(result: Any) -> Any:
    """Convert Enums and other non-serializable types to JSON-friendly primitives."""
    if isinstance(result, dict):
        return {k: _json_safe(v) for k, v in result.items()}
    if isinstance(result, list):
        return [_json_safe(v) for v in result]
    if hasattr(result, "value"):
        return str(result.value)
    return result


def _parse_step(step: Any) -> Dict[str, Any]:
    """
    Try to interpret a step payload into a dict form.
    Handles dicts, JSON strings, and Python repr strings.
    """
    if isinstance(step, dict):
        return step
    if isinstance(step, str):
        text = step.strip()
        if not text:
            return {}
        # Try JSON parse
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Try Python literal (for single-quoted dicts)
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _extract_files(steps: List[Any]) -> List[str]:
    """Extract and combine file contents from previous steps."""
    contents = []
    for step in steps:
        try:
            parsed = _parse_step(step)
            result = parsed.get("result", {})
            # result could be {'files': [...]} or a list of dicts
            if isinstance(result, dict) and "files" in result:
                for f in result["files"]:
                    if isinstance(f, dict) and "content" in f:
                        contents.append(f["content"])
            elif isinstance(result, list):
                for f in result:
                    if isinstance(f, dict) and "content" in f:
                        contents.append(f["content"])
        except Exception as e:
            logger.warning(f"[aggregate_file_content] Skipped step due to parse error: {e}")
            continue
    return contents[:MAX_FILES]


def run(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Aggregate file contents into a single string payload (<1MB total)."""
    params = params or {}
    steps = params.get("steps", [])
    contents = _extract_files(steps)

    if not contents:
        msg = "[aggregate_file_content] No file contents found to aggregate."
        logger.warning(msg)
        return _json_safe({
            "status": StepStatus.SUCCESS,
            "message": "Aggregated 0 file(s), total 0.0 KB.",
            "result": {"content": ""},
        })

    joined = "\n\n".join(contents)
    encoded = joined.encode("utf-8", errors="ignore")[:MAX_TOTAL_BYTES].decode("utf-8", errors="ignore")

    msg = f"Aggregated {len(contents)} file(s), total {len(encoded) / 1024:.1f} KB."
    logger.info(f"[aggregate_file_content] {msg}")

    return _json_safe({
        "status": StepStatus.SUCCESS,
        "message": msg,
        "result": {"content": encoded},
    })


def get_tool_definition():
    return {
        "name": "aggregate_file_content",
        "description": "Aggregates contents from multiple read_files results (max 20 files, 1MB total, JSON-safe).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": ["string", "object"]},
                    "description": "List of prior step outputs containing file contents.",
                }
            },
            "required": ["steps"],
        },
    }

