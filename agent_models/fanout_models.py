"""agent_models/fanout_models.py"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from agent_models.step_status import StepStatus  # your existing enum: SUCCESS/ERROR


class FanoutParams(BaseModel):
    """
    Minimal, permissive params model for fanout.
    - `root` is optional and may be injected/overridden per branch.
    - Extra keys are allowed (tool-specific params pass through untouched).
    """

    root: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # keep other tool params


class BranchOutput(BaseModel):
    """
    Minimal, permissive params model for fanout.
    root is optional and may be injected/overridden per branch.
    Extra tool-specific keys are allowed and passed through unchanged.

    Expected Input Example:
    {
    "root": "/home/me/proj", # optional; executor may override per branch
    "exclude": ["venv", "pycache"],
    "include": [".py", ".md"] # any additional tool params are permitted
    }

    Notes:

    If root is omitted, the executor sets it to <project_root>/<branch> during fan-out.

    Unknown/extra keys are retained (Config: extra="allow").

    * A branch’s execution record within a fan-out.

    Fields:
        branch (str): Branch identifier (usually a subpath relative to project root).
        status (StepStatus): Branch-specific tool status ("success" or "error").
        message (str): Optional branch-specific message (errors, summaries, etc.).
        output (Any): The tool’s branch-specific payload (the tool’s result field).

    >>> Expected Output Example:
    {
        "branch": "src",
        "status": "success",
        "message": "Found 5 files",
        "output": {
        "files": ["main.py", "utils.py", "config.yaml"]
        }
    }

    Notes:
    status accepts either StepStatus enum or string ("success"/"error"); strings are coerced.
    output is the tool’s raw result value for that branch (not re-wrapped).
    """

    branch: str
    status: StepStatus = StepStatus.SUCCESS
    message: str = ""
    output: Any = None

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v):
        """
        Accept "success"/"error" strings from tools and coerce to enum - StepStatus
        """
        if isinstance(v, str):
            s = v.lower().strip()
            if s == "success":
                return StepStatus.SUCCESS
            if s == "error":
                return StepStatus.ERROR
        return v


class FanoutPayload(BaseModel):
    """Container for per-branch results."""

    per_branch: List[BranchOutput] = Field(default_factory=list)


class FanoutResult(BaseModel):
    """
    Full merged fanout result with normalized envelope.
    This matches the executor’s `_fanout_over_branches` output structure, but typed.

    Expected Output Example:
    {
        "status": "success",            # overall status, 'error' if any branch failed
        "message": "",                   # aggregated error messages if any branch failed
        "result": {
            "per_branch": [
                {
                    "branch": "src",     # branch name (relative path from root)
                    "status": "success", # branch-specific tool status
                    "message": "Found 5 files",
                    "output": {          # branch-specific tool 'result' value
                        "files": ["main.py", "utils.py", "config.yaml"]
                    }
                },
                {
                    "branch": "tests",
                    "status": "success",
                    "message": "Found 3 files",
                    "output": {
                        "files": ["test_main.py", "test_utils.py"]
                    }
                }
            ]
        }
    }
    """

    status: StepStatus = StepStatus.SUCCESS
    message: str = ""
    result: FanoutPayload = Field(default_factory=FanoutPayload)

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v):
        """
        Accept "success"/"error" strings from tools and coerce to enum - StepStatus
        """
        if isinstance(v, str):
            s = v.lower().strip()
            if s == "success":
                return StepStatus.SUCCESS
            if s == "error":
                return StepStatus.ERROR
        return v
