# tools/read_files.py
"""
Reads file contents safely (up to 20 files <1MB each), MCP-compatible.
Now fully compatible with list_project_files (which returns JSON string under 'result').
Safely unwraps nested serialized dicts and ignores too-long filenames gracefully.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
from agent_models.step_status import StepStatus
import json
import ast
import logging

logger = logging.getLogger(__name__)

MAX_FILES = 20
MAX_FILE_SIZE = 1_000_000  # 1 MB
MAX_PATH_LENGTH = 255  # typical OS limit


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively convert Enums and other non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "value"):
        return str(obj.value)
    return obj


# ---------------------------------------------------------------------
# Path Extraction
# ---------------------------------------------------------------------
def _extract_paths(path_param: Union[str, list, dict]) -> List[str]:
    """
    Normalize and flatten input into a clean list of file paths.
    Handles:
      - dict with JSON string in 'result'
      - already parsed list
      - stringified Python dict or JSON string
    """
    # Case 1: list of paths
    if isinstance(path_param, list):
        return [str(p) for p in path_param if isinstance(p, (str, Path))]

    # Case 2: dict with 'result' (could be JSON string)
    if isinstance(path_param, dict):
        result = path_param.get("result")
        if isinstance(result, list):
            return [str(p) for p in result[:MAX_FILES]]
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    return [str(p) for p in parsed[:MAX_FILES]]
            except Exception as e:
                logger.warning(f"[read_files] Could not parse JSON result: {e}")
        return []

    # Case 3: string input — could be serialized dict
    if isinstance(path_param, str):
        text = path_param.strip()
        if not text:
            return []

        # Try JSON first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(p) for p in parsed[:MAX_FILES]]
            if isinstance(parsed, dict):
                return _extract_paths(parsed)
        except Exception:
            pass

        # Try literal_eval (Python dict repr)
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return _extract_paths(parsed)
        except Exception:
            pass

        logger.warning("[read_files] Could not decode path_param as JSON or dict.")
        return []

    return []


# ---------------------------------------------------------------------
# File Reading
# ---------------------------------------------------------------------
def _read_files_internal(paths: List[str]) -> Dict[str, Any]:
    """Open up to 20 small (<1MB) files and return their contents."""
    if not paths:
        return _json_safe({
            "status": StepStatus.ERROR,
            "message": "No valid file paths provided.",
            "result": {"files": []},
        })

    contents = []
    total_size = 0
    read_count = 0

    for p in paths[:MAX_FILES]:
        try:
            full_path = Path(p)

            if len(str(full_path)) > MAX_PATH_LENGTH:
                logger.warning(f"[read_files] Skipped too-long path: {str(full_path)[:100]}...")
                continue

            if not full_path.exists():
                logger.warning(f"[read_files] Skipped missing file: {full_path}")
                continue

            if full_path.is_dir():
                logger.warning(f"[read_files] Skipped directory: {full_path}")
                continue

            size = full_path.stat().st_size
            if size > MAX_FILE_SIZE:
                logger.warning(f"[read_files] Skipped large file (>1MB): {full_path}")
                continue

            with full_path.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                contents.append({"path": str(full_path), "content": content})
                total_size += size
                read_count += 1

        except Exception as e:
            logger.warning(f"[read_files] Error reading {p}: {e}")
            contents.append({"path": str(p), "error": str(e)})

    msg = f"Read {read_count} file(s) ({total_size/1024:.1f} KB total)."
    logger.info(f"[read_files] {msg}")

    return _json_safe({
        "status": StepStatus.SUCCESS,
        "message": msg,
        "result": {"files": contents},
    })


# ---------------------------------------------------------------------
# MCP Definition
# ---------------------------------------------------------------------
def get_tool_definition():
    return {
        "name": "read_files",
        "description": "Reads up to 20 files (<1MB each) and returns their contents (JSON-safe).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": ["string", "array", "object"],
                    "description": "Single path, list, or dict/string from list_project_files (with JSON string list under 'result').",
                }
            },
            "required": ["path"],
        },
    }


def run(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """MCP entrypoint."""
    params = params or {}
    paths = _extract_paths(params.get("path"))
    return _read_files_internal(paths)

