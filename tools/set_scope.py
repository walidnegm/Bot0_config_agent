"""tools/set_scope.py"""

from typing import List, Optional, Dict, Any
from agent_models.step_status import StepStatus


def set_scope(**kwargs) -> Dict[str, Any]:
    """
    No-op tool that tells the executor what scope to use.
    - root: absolute or ~-expanded base directory
    - branches: list[str] relative to root

    Output is deliberately simple; executor reads these keys.
    """
    root: Optional[str] = kwargs.get("root")
    branches: List[str] = kwargs.get("branches") or []

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Scope set (root={root!r}, branches={branches})",
        "result": {"root": root, "branches": branches},
    }
