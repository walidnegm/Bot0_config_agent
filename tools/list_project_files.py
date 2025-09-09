# tools/list_project_files.py
import os
import logging
from typing import Optional, List
from pathlib import Path

from utils.find_root_dir import find_project_root
from tools.tool_models import ListProjectFilesOutput
from agent_models.step_status import StepStatus

logger = logging.getLogger(__name__)

def normalize_extension(ext: str) -> str:
    ext = ext.strip()
    if ext.startswith("*."):
        ext = ext[1:]
    if not ext.startswith("."):
        ext = "." + ext.lstrip(".")
    return ext

def list_project_files(
    root: Optional[str] = None,
    include: Optional[List[str]] = None,
    include_file: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    exclude_dir: Optional[List[str]] = None
) -> ListProjectFilesOutput:
    include_raw = include or include_file or [".py", ".md", ".toml", ".yaml", ".json"]
    if isinstance(include_raw, str):
        include_raw = [ext.strip() for ext in include_raw.split(",") if ext.strip()]
    include = [normalize_extension(ext) for ext in include_raw]

    logger.debug(f"[list_project_files] Using include filter: {include}")

    MINIMUM_EXCLUDE = {".venv", "venv", "__pycache__", ".git", "models", "node_modules", "sandbox"}
    exclude_set = set(exclude or exclude_dir or [])
    exclude_set |= MINIMUM_EXCLUDE

    root_path = root or find_project_root()
    if not root:
        logger.warning("[list_project_files] No 'root' specified; defaulting to project root.")
    root_path = Path(root_path) if isinstance(root_path, str) else root_path
    if not root_path.exists():
        return ListProjectFilesOutput(status=StepStatus.ERROR, message=f"Directory '{root_path}' does not exist.", result=None)

    logger.info(f"[list_project_files] Scanning directory: {root_path}")

    file_paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in exclude_set]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix in include:
                try:
                    if file_path.stat().st_size <= 10_000:
                        file_paths.append(str(file_path))
                except OSError:
                    continue

    logger.info(f"Found {len(file_paths)} files under {root_path}.")
    return ListProjectFilesOutput(status=StepStatus.SUCCESS, message=f"Found {len(file_paths)} file(s) under '{root_path}'", result=file_paths)
