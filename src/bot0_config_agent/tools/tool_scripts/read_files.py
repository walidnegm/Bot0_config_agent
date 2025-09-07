"""bot0_config_agent/tools/tool_scripts/read_files.py"""

from pathlib import Path
from typing import List
from bot0_config_agent.agent_models.step_status import StepStatus


def read_files(**kwargs):
    """Tool to read one or more files from string or Path inputs.

    Params:
        files (list[str|Path] | str | Path): File paths to read.

    Returns:
        dict: {
            "status": StepStatus,
            "message": str,
            "result": {"files": [{"file": str, "content": str}, ...]} | None
        }
    """
    files = kwargs.get("files")
    if not files:
        return {
            "status": StepStatus.ERROR,
            "message": "Missing required parameter: files",
            "result": None,
        }

    # Normalize to list
    if isinstance(files, (str, Path)):
        files = [files]
    elif not isinstance(files, list):
        return {
            "status": StepStatus.ERROR,
            "message": "Invalid type for 'files'; expected str | Path | list[str|Path].",
            "result": None,
        }

    contents: List[dict] = []
    for file in files:
        try:
            file_path = Path(file)
            with file_path.open("r", encoding="utf-8") as f:
                contents.append({"file": str(file_path), "content": f.read()})
        except Exception as e:
            return {
                "status": StepStatus.ERROR,
                "message": f"Failed to read file '{file}': {e}",
                "result": None,
            }

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Read {len(contents)} file(s)",
        "result": {"files": contents},
    }
