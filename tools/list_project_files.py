"""tools/list_project_files.py"""

import os
import logging
from typing import Any, Dict
from pathlib import Path
from utils.find_root_dir import find_project_root


logger = logging.getLogger(__name__)


def list_project_files(**kwargs) -> Dict[str, Any]:
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
        dict: {
            "status": "success" or "error",
            "files": List[str],  # List of file paths as strings
            "message": str       # Summary of results or error message
        }

    Notes:
        - Excluded directories are never traversed (efficient pruning).
        - Only files whose suffix matches an entry in 'include' are considered.
        - Only files <= 10,000 bytes are included in the result.
    """

    root = kwargs.get("root") or find_project_root()
    if isinstance(root, str):
        root = Path(root)
    if not root.exists():
        return {"status": "error", "message": f"Directory '{root}' does not exist."}

    logger.info(f"[list_project_files] 🔍 Scanning directory: {root}")

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

    # Prepare include filter (always with a leading '.')
    include = [
        ext if ext.startswith(".") else f".{ext.lstrip('.')}"
        for ext in (
            kwargs.get("include")
            or kwargs.get("include_file")
            or [".py", ".md", ".toml", ".yaml", ".json"]
        )
    ]

    file_paths = []
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
    return {
        "status": "success",
        "files": file_paths,
        "message": f"Found {len(file_paths)} file(s) under '{root}'",
    }
