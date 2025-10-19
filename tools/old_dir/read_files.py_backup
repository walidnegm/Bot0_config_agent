"""tools/read_files.py"""

from pathlib import Path
from agent_models.step_status import StepStatus


def read_files(**kwargs):
    """Tool to read one or more files from string or Path inputs."""
    paths = kwargs.get("path")
    contents = []

    if not paths:
        return {
            "status": StepStatus.ERROR,
            "message": "Missing required parameter: path",
        }

    # Normalize to list
    if isinstance(paths, (str, Path)):
        paths = [paths]
    elif not isinstance(paths, list):
        return {"status": StepStatus.ERROR, "message": "Invalid path type"}

    for path in paths:
        try:
            path = Path(path)  # Ensure Path object
            with path.open("r", encoding="utf-8") as f:
                contents.append({"path": str(path), "content": f.read()})
        except Exception as e:
            return {
                "status": StepStatus.ERROR,
                "message": f"Failed to read file '{path}': {e}",
            }

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Read {len(contents)} file(s)",
        "result": {"files": contents},
    }
