"""
tools/find_dir_size.py

Tool to generate file size & number of files stats for a directory.
"""

from pathlib import Path
import humanize
from agent_models.step_status import StepStatus


def find_dir_size(**kwargs):
    """
    Find the file size of a directory and number of files.
    """
    root = kwargs.get("root") or "."
    path = Path(root)
    if not path.exists() or not path.is_dir():
        return {
            "status": StepStatus.SUCCESS,
            "message": f"Directory '{root}' does not exist or is not a directory.",
        }

    total_size = 0
    num_files = 0
    seen_inodes = set()

    for file in path.rglob("*"):
        if file.is_file() and not file.is_symlink():
            try:
                stat = file.stat()
                # Track inode to avoid double-counting hard links
                if stat.st_ino in seen_inodes:
                    continue
                seen_inodes.add(stat.st_ino)
                num_files += 1
                total_size += stat.st_size
            except Exception:
                pass  # Ignore unreadable files

    # Format in GB, MB, etc.
    size_hr = humanize.naturalsize(total_size, binary=True)

    return {
        "status": StepStatus.SUCCESS,
        "message": f"{num_files} files, {size_hr} in '{root}'",
        "result": {
            "num_files": num_files,
            "total_size_bytes": total_size,
            "total_size_hr": size_hr,
            "root": str(path.resolve()),
        },
    }
