"""bot0_config_agent/tools/tool_scripts/set_scope.py"""

from typing import List, Optional, Dict, Any
from bot0_config_agent.agent_models.step_status import StepStatus


def set_scope(**kwargs) -> Dict[str, Any]:
    """
    No-op tool that tells the executor what scope to use.
    - dir: absolute or ~-expanded base directory
    - branches: list[str] relative to root

    Output is deliberately simple; executor reads these keys.
    """
    dir_path: Optional[str] = kwargs.get("dir")
    branches: List[str] = kwargs.get("branches") or []

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Scope set (dir={dir_path!r}, branches={branches})",
        "result": {"dir": dir_path, "branches": branches},
    }
