"""
tools/check_disk_usage.py
MCP tool: reports disk usage stats for the current system.
"""

import shutil
import platform

def main(path: str = "/") -> dict:
    """Return total, used, and free disk space (in GB) for the given path."""
    total, used, free = shutil.disk_usage(path)
    return {
        "status": "success",
        "message": f"Disk usage for {path} on {platform.system()}",
        "result": {
            "path": path,
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "percent_used": round(used / total * 100, 1),
        },
    }

# ✅ MCP servers recognize this callable
def get_tool_definition():
    return {
        "name": "check_disk_usage",
        "description": "Reports total, used, and free disk space for a given path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Filesystem path to analyze"},
            },
            "required": [],
        },
    }

# ✅ Entry point exposed to MCP discovery
def run(params):
    path = params.get("path", "/")
    return main(path)

