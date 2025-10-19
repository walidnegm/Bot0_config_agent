# tools/echo_message.py
"""
Echoes a message back to the caller.
Now MCP-compatible: defines `get_tool_definition()` and `run(params)`.
"""

import json
from typing import Dict, Any
from agent_models.step_status import StepStatus


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def _echo_message_internal(message: str) -> Dict[str, Any]:
    """Echo the given message and attempt to parse JSON if applicable."""
    result = {
        "status": StepStatus.SUCCESS,
        "message": message,
        "result": {"echo": message},
    }

    try:
        parsed = json.loads(message)
        if isinstance(parsed, dict):
            result["result"]["parsed"] = parsed
    except Exception:
        # Not JSON — keep raw message
        pass

    return result


# ---------------------------------------------------------------------
# MCP entry points
# ---------------------------------------------------------------------
def get_tool_definition():
    return {
        "name": "echo_message",
        "description": "Echoes a message or JSON string back to the caller.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message or JSON string to echo back.",
                }
            },
            "required": ["message"],
        },
    }


def run(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """MCP entrypoint wrapper."""
    params = params or {}
    message = params.get("message", "")
    return _echo_message_internal(message)

