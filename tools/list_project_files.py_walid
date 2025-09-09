"""tools/list_project_files.py"""

import os
import logging
from typing import Any, Dict, List
from pathlib import Path

from utils.find_root_dir import find_project_root
from tools.tool_models import ListProjectFilesOutput
from agent_models.step_status import StepStatus

logger = logging.getLogger(__name__)

def normalize_extension(ext: str) -> str:
    # Handles .py, py, *.py, *py, etc → .py
    ext = ext.strip()
    if ext.startswith("*."):
        ext = ext[1:]  # remove '*'
    if not ext.startswith("."):
        ext = "." + ext.lstrip(".")
    return ext

def list_project_files(**kwargs) -> ListProjectFilesOutput:
    """
    Recursively scans a project directory for files matching given criteria,
    with efficient directory pruning and file-type exclusion using os.walk.

    Args:
        root (str or Path, optional): The root directory to scan.
            If not provided, uses the project root (via find_project_root()).
        exclude (list of str, optional): Directory names to exclude from scanning
            (e.g., ['venv', '__pycache__', 'models', '.git']).
        exclude_dir (list of str, optional): Alternate key for 'exclude'.
        include (list of str, optional): File extensions to include (e.g.,
            ['.py', '.md', '.toml', '.yaml', '.json']).
        include_file (list of str, optional): Alternate key for 'include'.

    Returns:
        ListProjectFilesOutput:
            - status: success | error
            - message: summary string
            - result: List[str] (file paths)
    """
    # Prepare include filter (always with a leading '.')
    include: List[str] = [
        normalize_extension(ext)
        for ext in (
            kwargs.get("include")
            or kwargs.get("include_file")
            or [".py", ".md", ".toml", ".yaml", ".json"]
        )
    ]
    logger.debug(f"[list_project_files] Using include filter: {include}")

    # Define a robust default exclusion set to avoid LLM mistakes
    MINIMUM_EXCLUDE = {
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        "models",
        "node_modules",
        "sandbox",
    }
    # Union of user-supplied and minimum exclusions
    exclude = set(kwargs.get("exclude") or kwargs.get("exclude_dir") or [])
    exclude |= MINIMUM_EXCLUDE

    root = kwargs.get("root") or find_project_root()
    if not kwargs.get("root"):
        logger.warning("[list_project_files] No 'root' specified; defaulting to project root. Ensure planner sets 'root' for subdirectory-specific tasks.")
    if isinstance(root, str):
        root = Path(root)
    if not root.exists():
        return ListProjectFilesOutput(
            status=StepStatus.ERROR,
            message=f"Directory '{root}' does not exist.",
            result=None,
        )

    logger.info(f"[list_project_files] 🔍 Scanning directory: {root}")

    file_paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place (most efficient way)
        before = set(dirnames)
        dirnames[:] = [d for d in dirnames if d not in exclude]
        pruned = before - set(dirnames)
        if pruned:
            for skipped in pruned:
                logger.debug(f"Pruned directory: {Path(dirpath) / skipped}")

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix in include:
                try:
                    if file_path.stat().st_size <= 10_000:
                        file_paths.append(str(file_path))
                except OSError:
                    continue

    logger.info(f"Found {len(file_paths)} files under {root}.")
    return ListProjectFilesOutput(
        status=StepStatus.SUCCESS,
        message=f"Found {len(file_paths)} file(s) under '{root}'",
        result=file_paths
    )
