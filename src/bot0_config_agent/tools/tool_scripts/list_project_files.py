"""bot0_config_agent/tools/tool_scripts/list_project_files.py"""

import os
import logging
from typing import Any, Dict, List
from pathlib import Path
from bot0_config_agent.utils.system.find_root_dir import find_project_root
from bot0_config_agent.agent_models.step_status import StepStatus

logger = logging.getLogger(__name__)


def normalize_extension(ext: str) -> str:
    """Normalize extensions like .py / py / *.py / *py → .py"""
    ext = ext.strip()
    if ext.startswith("*."):
        ext = ext[1:]  # drop leading '*'
    if not ext.startswith("."):
        ext = "." + ext.lstrip(".")
    return ext


def list_project_files(**kwargs) -> Dict[str, Any]:
    """
    Recursively scan a project directory for files, with efficient directory pruning
    and extension filtering.

    Args:
        dir (str | Path, optional): Root directory to scan. Defaults to project root.
        exclude (list[str], optional): Directory names to exclude (e.g., ['venv', '__pycache__']).
        exclude_dir (list[str], optional): Alias for 'exclude'.
        include (list[str], optional): File extensions to include (e.g., ['.py', '.md']).
        include_file (list[str], optional): Alias for 'include'.

    Returns:
        dict: Standard tool envelope:
            {
              "status": StepStatus.SUCCESS | StepStatus.ERROR,
              "message": str,
              "result": { "files": List[str] }
            }

    Notes:
        - Excluded directories are never traversed (pruned in-place).
        - Only files whose suffix matches an entry in 'include' are returned.
        - Files larger than 10_000 bytes are skipped.
    """
    # Build include filter (normalized to leading '.')
    include: List[str] = [
        normalize_extension(ext)
        for ext in (
            kwargs.get("include")
            or kwargs.get("include_file")
            or [".py", ".md", ".toml", ".yaml", ".json"]
        )
    ]
    logger.debug(f"[list_project_files] Using include filter: {include}")

    # Minimum exclusions to avoid common noise
    MINIMUM_EXCLUDE = {
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        "models",
        "node_modules",
        "sandbox",
    }
    exclude = set(kwargs.get("exclude") or kwargs.get("exclude_dir") or [])
    exclude |= MINIMUM_EXCLUDE

    # Resolve root dir
    root = kwargs.get("dir") or find_project_root()
    root_path = Path(root) if not isinstance(root, Path) else root

    if not root_path.exists() or not root_path.is_dir():
        return {
            "status": StepStatus.ERROR,
            "message": f"Directory '{root_path}' does not exist or is not a directory.",
            "result": {"files": []},
        }

    logger.info(f"[list_project_files] 🔍 Scanning directory: {root_path}")

    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune excluded directories in-place
        before = set(dirnames)
        dirnames[:] = [d for d in dirnames if d not in exclude]
        pruned = before - set(dirnames)
        for skipped in pruned:
            logger.debug(
                f"[list_project_files] Pruned directory: {Path(dirpath) / skipped}"
            )

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix in include:
                try:
                    if file_path.stat().st_size <= 10_000:
                        files.append(str(file_path))
                except OSError:
                    # unreadable file; skip
                    continue

    logger.info(f"[list_project_files] Found {len(files)} files under {root_path}.")
    return {
        "status": StepStatus.SUCCESS,
        "message": f"Found {len(files)} file(s) under '{root_path}'",
        "result": {"files": files},
    }
