# tools/read_files.py
from pathlib import Path
from typing import List, Dict, Any
from agent_models.step_status import StepStatus

def read_files(path: List[str]) -> Dict[str, Any]:
    contents = []

    if not path:
        return {"status": StepStatus.ERROR, "message": "Missing required parameter: path"}

    for p in path:
        try:
            full_path = Path(p)
            with full_path.open("r", encoding="utf-8") as f:
                contents.append({"path": str(full_path), "content": f.read()})
        except Exception as e:
            return {"status": StepStatus.ERROR, "message": f"Failed to read file '{p}': {e}"}

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Read {len(contents)} file(s)",
        "result": {"files": contents},
    }
