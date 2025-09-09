"""
toos/find_dir_structure.py

Tool to recursive find sub directories and return a dict of directory/file structure.

"""

# tools/find_dir_structure.py
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


def _build_structure(path: Path, ignore_dirs: List[str]) -> Dict[str, Any]:
    """Recursive helper to build directory tree."""
    structure: Dict[str, Any] = {
        "name": path.name,
        "type": "directory" if path.is_dir() else "file",
    }
    if path.is_dir():
        children = []
        for item in sorted(path.iterdir()):
            if item.name in ignore_dirs:
                continue
            children.append(_build_structure(item, ignore_dirs))
        structure["children"] = children
    return structure


def _count_stats(structure: Dict[str, Any]) -> Dict[str, int]:
    """Count files and directories in a tree dict."""
    files = 0
    dirs = 0
    if structure["type"] == "file":
        return {"files": 1, "dirs": 0}
    dirs += 1
    for child in structure.get("children", []):
        c = _count_stats(child)
        files += c["files"]
        dirs += c["dirs"]
    return {"files": files, "dirs": dirs}


def find_dir_structure(
    path: Union[Path, str], ignore_dirs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Return hierarchical directory structure with summary message."""
    path = Path(path).resolve()
    if ignore_dirs is None:
        ignore_dirs = [".git", "__pycache__", "node_modules", ".venv"]

    if not path.exists():
        return {
            "status": "error",
            "message": f"Directory '{path}' does not exist.",
            "result": None,
        }

    try:
        structure = _build_structure(path, ignore_dirs)
        stats = _count_stats(structure)
        message = f"Found {stats['files']} files and {stats['dirs']} directories under '{path.name}'."
        return {
            "status": "success",
            "message": message,
            "result": structure,
        }
    except (FileNotFoundError, PermissionError) as e:
        return {
            "status": "error",
            "message": f"Error accessing '{path}': {e}",
            "result": None,
        }
