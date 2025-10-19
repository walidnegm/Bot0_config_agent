# tools/list_project_files.py
"""
Lists project files under a given directory with filtering options.
Now fully MCP-compatible and JSON-safe: the 'result' field is a JSON string list
so that downstream tools (read_files, aggregate_file_content) can safely parse it.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from agent_models.step_status import StepStatus
from utils.find_root_dir import find_project_root

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def normalize_extension(ext: str) -> str:
    ext = ext.strip()
    if ext.startswith("*."):
        ext = ext[1:]
    if not ext.startswith("."):
        ext = "." + ext.lstrip(".")
    return ext


def _json_safe(obj: Any) -> Any:
    """Recursively convert enums and non-serializable objects to JSON primitives."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "value"):  # e.g. StepStatus.SUCCESS
        return str(obj.value)
    return obj


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def _list_files(
    root: Optional[str],
    include: Optional[List[str]],
    exclude: Optional[List[str]],
) -> Dict[str, Any]:
    """List project files given filters."""
    include_raw = include or [".py", ".md", ".toml", ".yaml", ".json"]
    if isinstance(include_raw, str):
        include_raw = [ext.strip() for ext in include_raw.split(",") if ext.strip()]
    include = [normalize_extension(ext) for ext in include_raw]

    MINIMUM_EXCLUDE = {
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        "models",
        "node_modules",
        "sandbox",
    }
    exclude_set = set(exclude or []) | MINIMUM_EXCLUDE

    root_path = root or find_project_root()
    if not root:
        logger.warning("[list_project_files] No 'root' specified; defaulting to project root.")
    root_path = Path(root_path)
    if not root_path.exists():
        return _json_safe({
            "status": StepStatus.ERROR,
            "message": f"Directory '{root_path}' does not exist.",
            "result": "[]",  # ✅ return valid JSON list string
        })

    logger.info(f"[list_project_files] Scanning directory: {root_path}")

    file_paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in exclude_set]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix in include:
                try:
                    # Limit large or binary files
                    if file_path.stat().st_size <= 10_000:
                        file_paths.append(str(file_path))
                except OSError:
                    continue

    msg = f"Found {len(file_paths)} file(s) under '{root_path}'"
    logger.info(f"[list_project_files] {msg}")

    # ✅ result is now an explicit JSON array string, never a Python repr
    return _json_safe({
        "status": StepStatus.SUCCESS,
        "message": msg,
        "result": json.dumps(file_paths),
    })


# ---------------------------------------------------------------------
# MCP entry points
# ---------------------------------------------------------------------
def get_tool_definition():
    return {
        "name": "list_project_files",
        "description": "Lists project files with optional include/exclude filters (JSON-safe).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Root directory to start scanning (defaults to project root).",
                },
                "include": {
                    "type": ["array", "string"],
                    "items": {"type": "string"},
                    "description": "File extensions to include, e.g., ['.py', '.json'].",
                },
                "exclude": {
                    "type": ["array", "string"],
                    "items": {"type": "string"},
                    "description": "Directories or patterns to exclude.",
                },
            },
            "required": [],
        },
    }


def run(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """MCP entrypoint."""
    params = params or {}
    root = params.get("root")
    include = params.get("include")
    exclude = params.get("exclude")
    return _list_files(root=root, include=include, exclude=exclude)

