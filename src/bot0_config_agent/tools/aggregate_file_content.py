"""
tools/aggregate_file_content.py

Combine multiple file contents from prior steps for summarization
"""

from typing import Any, List
from bot0_config_agent.agent_models.step_status import StepStatus


def _extract_text_blocks(items: List[Any]) -> List[str]:
    """
    Best-effort extractor that turns a heterogeneous list of step outputs into text blocks.
    Handles common shapes:
      - plain strings/bytes
      - dicts with "result"
      - read_files-like payloads: {"result": {"files": [{"file": "...", "content": "..."}]}}
    Falls back to str(...) for anything else.
    """
    blocks: List[str] = []
    for x in items:
        if x is None:
            continue

        # bytes → str
        if isinstance(x, (bytes, bytearray)):
            blocks.append(x.decode(errors="replace"))
            continue

        # dict shapes (common for tool outputs)
        if isinstance(x, dict):
            r = x.get("result", x)  # prefer the payload if present
            # read_files-like: {"files": [{"file": "...", "content": "..."}]}
            if isinstance(r, dict) and "files" in r and isinstance(r["files"], list):
                for it in r["files"]:
                    if isinstance(it, dict) and "content" in it:
                        blocks.append(str(it["content"]))
                    else:
                        blocks.append(str(it))
                continue

            # direct string result
            if isinstance(r, str):
                blocks.append(r)
                continue

            # generic dict payload → stringify
            blocks.append(str(r))
            continue

        # plain string or other primitive
        if isinstance(x, str):
            blocks.append(x)
        else:
            blocks.append(str(x))

    return blocks


def aggregate_file_content(**kwargs):
    """
    Aggregate multiple file contents into a single joined string.

    This tool expects a list of file contents from prior steps (e.g., outputs of
    `read_files` or `summarize_files`) under the `steps` parameter. It concatenates
    them with double newlines between each file.

    Returns:
        dict: Standard tool result envelope:
            {
                "status": StepStatus,   # SUCCESS or ERROR
                "message": str,         # Summary of the outcome
                "result": str | None,   # Joined content or None on error
            }

    Example:
        >>> aggregate_file_content(steps=["file1 content", "file2 content"])
        {
            "status": StepStatus.SUCCESS,
            "message": "Aggregated file contents.",
            "result": "file1 content\n\nfile2 content",
        }
    """
    steps = kwargs.get("steps", [])  # Expected resolved step values

    if not steps or not isinstance(steps, list):
        return {
            "status": StepStatus.ERROR,
            "message": "Missing or invalid 'steps' parameter. Must be a list of file contents.",
            "result": None,
        }

    try:
        joined = "\n\n".join(str(s) for s in steps)
        return {
            "status": StepStatus.SUCCESS,
            "message": "Aggregated file contents.",
            "result": joined,
        }
    except Exception as e:
        return {
            "status": StepStatus.ERROR,
            "message": f"Failed to aggregate: {e}",
            "result": None,
        }
