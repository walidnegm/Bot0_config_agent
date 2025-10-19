# tools/check_python_version.py
"""
Reports Python version and environment details.
Now MCP-compatible: defines `get_tool_definition()` and `run(params)`.
"""

import sys
import platform
from typing import Dict, Any
from agent_models.step_status import StepStatus


# ---------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------
def _get_python_info() -> Dict[str, Any]:
    """Return system and interpreter details."""
    try:
        version_info = {
            "python_version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": platform.python_build(),
            "platform": platform.platform(),
        }
        return {
            "status": StepStatus.SUCCESS,
            "message": f"Python {version_info['python_version']} detected.",
            "result": version_info,
        }
    except Exception as e:
        return {"status": StepStatus.ERROR, "message": str(e), "result": None}


# ---------------------------------------------------------------------
# MCP entry points
# ---------------------------------------------------------------------
def get_tool_definition():
    return {
        "name": "check_python_version",
        "description": "Reports the Python interpreter version and environment information.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def run(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """MCP entrypoint — params ignored (no input needed)."""
    return _get_python_info()

