"""
MCP tool: reports total size and file count of a directory (recursively).
"""

from pathlib import Path
import humanize


def main(root: str = ".") -> dict:
    """Return total file count and size for a given directory."""
    path = Path(root)
    if not path.exists() or not path.is_dir():
        return {
            "status": "error",
            "message": f"Directory '{root}' does not exist or is not a directory.",
            "result": {},
        }

    total_size = 0
    num_files = 0
    seen_inodes = set()

    for file in path.rglob("*"):
        if file.is_file() and not file.is_symlink():
            try:
                stat = file.stat()
                if stat.st_ino in seen_inodes:
                    continue
                seen_inodes.add(stat.st_ino)
                num_files += 1
                total_size += stat.st_size
            except Exception:
                continue

    size_hr = humanize.naturalsize(total_size, binary=True)

    return {
        "status": "success",
        "message": f"{num_files} files, {size_hr} in '{root}'",
        "result": {
            "num_files": num_files,
            "total_size_bytes": total_size,
            "total_size_hr": size_hr,
            "root": str(path.resolve()),
        },
    }


# ✅ MCP metadata definition
def get_tool_definition():
    return {
        "name": "find_dir_size",
        "description": "Reports total number of files and total size (in bytes and human-readable form) of a directory recursively.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Directory path to analyze (default: current directory)",
                },
            },
            "required": [],
        },
    }


# ✅ Entry point exposed to MCP discovery
def run(params):
    root = params.get("root", ".")
    return main(root)

