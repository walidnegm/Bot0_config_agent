"""
tools/find_dir_size.py

Tool to compute the total size and number of files in a directory.
"""

from pathlib import Path
import humanize
from bot0_config_agent.agent_models.step_status import StepStatus


def find_dir_size(**kwargs):
    """
    Compute file statistics for a given directory, including the number of files
    and their cumulative size.

    Args:
        **kwargs:
            dir (str, optional): Path to the directory to scan. Defaults to ".".

    Returns:
        dict: Standardized result in the format:
            {
                "status": StepStatus,
                "message": str,
                "result": {
                    "num_files": int,
                    "total_size_bytes": int,
                    "total_size_hr": str,
                    "root": str
                }
            }
    """
    dir = kwargs.get("dir") or "."
    path = Path(dir)
    if not path.exists() or not path.is_dir():
        return {
            "status": StepStatus.ERROR,
            "message": f"Directory '{dir}' does not exist or is not a directory.",
            "result": {},
        }

    total_size = 0
    num_files = 0
    seen_inodes = set()

    for file in path.rglob("*"):
        if file.is_file() and not file.is_symlink():
            try:
                stat = file.stat()
                # Avoid double-counting hard links
                if stat.st_ino in seen_inodes:
                    continue
                seen_inodes.add(stat.st_ino)
                num_files += 1
                total_size += stat.st_size
            except Exception:
                # Ignore unreadable files
                pass

    size_hr = humanize.naturalsize(total_size, binary=True)

    return {
        "status": StepStatus.SUCCESS,
        "message": f"{num_files} files, {size_hr} in '{dir}'",
        "result": {
            "num_files": num_files,
            "total_size_bytes": total_size,
            "total_size_hr": size_hr,
            "root": str(path.resolve()),
        },
    }
