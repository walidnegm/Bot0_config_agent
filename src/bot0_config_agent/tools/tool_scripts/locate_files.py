"""
bot0_config_agent/tools/tool_scripts/locate_file.py

Locate a file by exact name under common roots.
"""

import os
from typing import Dict, Any, List, Optional
from bot0_config_agent.agent_models.step_status import StepStatus


def locate_files(**kwargs) -> Dict[str, Any]:
    """
    Search for an exact filename under common roots (home dir and CWD).

    Args (kwargs):
        filename (str): Exact filename to locate (e.g., "config.yaml").

    Returns:
        dict: Standard tool envelope (status, message, result)
              - status: StepStatus.SUCCESS | StepStatus.ERROR
              - message: summary string
              - result: {
                    "path": Optional[str],      # first match (or None)
                    "all_matches": List[str]    # all matches (may be empty)
                }
    """
    filename: Optional[str] = kwargs.get("filename")
    if not filename or not isinstance(filename, str):
        return {
            "status": StepStatus.ERROR,
            "message": "Missing required parameter: 'filename' (str).",
            "result": None,
        }

    matches: List[str] = []
    search_roots = [os.path.expanduser("~"), os.getcwd()]

    for root in search_roots:
        for dirpath, _, files in os.walk(root):
            # exact filename match only
            if filename in files:
                matches.append(os.path.join(dirpath, filename))

    first_match = matches[0] if matches else None
    return {
        "status": StepStatus.SUCCESS,
        "message": f"Found {len(matches)} match(es) for '{filename}'.",
        "result": {"path": first_match, "all_matches": matches},
    }
