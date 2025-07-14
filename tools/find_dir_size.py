"""
tools/find_dir_size.py

Tool to generate file size & number of files stats for a directory.
"""

from pathlib import Path


def find_dir_size(**kwargs):
    root = kwargs.get("root") or "."
    path = Path(root)
    if not path.exists() or not path.is_dir():
        return {
            "status": "error",
            "message": f"Directory '{root}' does not exist or is not a directory.",
        }

    total_size = 0
    num_files = 0

    for file in path.rglob("*"):
        if file.is_file():
            num_files += 1
            try:
                total_size += file.stat().st_size
            except Exception:
                pass  # Ignore unreadable files

    size_mb = total_size / (1024 * 1024)
    return {
        "status": "ok",
        "result": {
            "num_files": num_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(size_mb, 2),
            "root": str(path.resolve()),
        },
        "message": f"{num_files} files, {round(size_mb,2)} MB in '{root}'",
    }
