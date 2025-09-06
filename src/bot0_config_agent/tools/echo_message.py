# tools/echo_message.py
# ---------------------
# Simple echo tool with optional JSON pretty-print.

import json
from bot0_config_agent.agent_models.step_status import StepStatus


def echo_message(**kwargs):
    """
    Echo the provided message back to the caller.

    Args:
        message (str): The message to echo. If the message is a JSON string,
                       it will be pretty-printed in the result. If that JSON
                       contains a top-level "matches" list, the step message
                       will note the count.

    Returns:
        dict: Standard tool envelope (status, message, result)
              - status: StepStatus.SUCCESS | StepStatus.ERROR
              - message: short summary (e.g., "Echoed message (N match(es)).")
              - result: the echoed content (string; pretty-printed if JSON)
    """
    msg = kwargs.get("message", "")

    # Default envelope pieces
    status = StepStatus.SUCCESS
    summary_note = ""
    result_str = msg

    # Try to pretty-print JSON messages and surface matches count in summary
    try:
        parsed = json.loads(msg)
        if isinstance(parsed, (dict, list)):
            result_str = json.dumps(parsed, ensure_ascii=False, indent=2)
            if isinstance(parsed, dict) and isinstance(parsed.get("matches"), list):
                summary_note = f" ({len(parsed['matches'])} match(es))"
    except Exception:
        # Not JSON or failed to parse → return raw string
        pass

    return {
        "status": status,
        "message": f"Echoed message{summary_note}.",
        "result": result_str,
    }
