"""bot0_config_agent/tools/tool_scripts/seed_parser.py

Parse simple KEY=VALUE pairs from a plaintext seed/config file (.env-style).
"""

from pathlib import Path
from typing import Dict
from bot0_config_agent.agent_models.step_status import StepStatus


def _parse_kv_lines(text: str) -> Dict[str, str]:
    """
    Minimal .env-style parser:
      - Ignores empty lines and comments (# or ;)
      - Supports optional 'export ' prefix
      - Uses the last occurrence of a duplicated key
      - Trims surrounding quotes from values
    """
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].strip()

        if "=" not in s:
            continue

        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def seed_parser(**kwargs):
    """
    Params (kwargs):
        file (str | Path): Path to a plaintext file containing KEY=VALUE lines.

    Returns:
        dict (status, message, result):
            {
              "status": StepStatus,
              "message": str,
              "result": {"parsed_data": dict[str, str]} | None
            }
    """
    file_arg = kwargs.get("file")
    if not file_arg:
        return {
            "status": StepStatus.ERROR,
            "message": "Missing required parameter: file",
            "result": None,
        }

    try:
        file_path = Path(file_arg)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        kv_pairs = _parse_kv_lines(text)
        return {
            "status": StepStatus.SUCCESS,
            "message": f"Parsed {len(kv_pairs)} key(s) from '{file_path}'.",
            "result": {"parsed_data": kv_pairs},
        }
    except Exception as e:
        return {
            "status": StepStatus.ERROR,
            "message": f"Failed to parse '{file_arg}': {e}",
            "result": None,
        }
