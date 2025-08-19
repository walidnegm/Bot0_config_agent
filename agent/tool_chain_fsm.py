"""
agent/tool_chain_fsm.py
Lite FSM controller to help execute 'chain of tools'.
"""

from typing import Dict, Any, Optional
from agent_models.agent_models import ToolChain
from agent_models.step_state import StepState


class ToolChainFSM:
    """
    FSM controller for executing a validated multi-step ToolChain plan.

    - Tracks each step's execution state.
    - Enforces that a step can only run if dependencies (previous steps) are completed.
    - Provides access to step data for the ToolChainExecutor.
    """

    def __init__(self, plan: ToolChain):
        """
        Args:
            plan (ToolChain): A validated ToolChain model with resolved parameters.
        """
        if not isinstance(plan, ToolChain):
            raise TypeError("ToolChainFSM requires a validated ToolChain instance.")

        self.plan = plan
        self.state_map: Dict[str, Dict[str, Any]] = {}
        self._initialize_states()

    def _initialize_states(self):
        """Initialize all steps in the plan as pending."""
        for i, _ in enumerate(self.plan.steps):
            self.state_map[f"step_{i}"] = {"state": StepState.PENDING, "result": None}

    def get_next_pending_step(self) -> Optional[str]:
        """
        Return the next step ID that is ready to run, or None if none are ready.
        A step is ready if it is pending and:
          - It is step_0
          - OR the previous step is completed
        """
        for i in range(len(self.plan.steps)):
            step_id = f"step_{i}"
            if self.state_map[step_id]["state"] == StepState.PENDING:
                if (
                    i == 0
                    or self.state_map[f"step_{i-1}"]["state"] == StepState.COMPLETED
                ):
                    return step_id
        return None

    def mark_in_progress(self, step_id: str):
        """Mark a step as currently executing."""
        self.state_map[step_id]["state"] = StepState.IN_PROGRESS

    def mark_completed(self, step_id: str, result: Any):
        """Mark a step as completed and store its result."""
        self.state_map[step_id]["state"] = StepState.COMPLETED
        self.state_map[step_id]["result"] = result

    def mark_failed(self, step_id: str, error: str):
        """Mark a step as failed and store the error message."""
        self.state_map[step_id]["state"] = StepState.FAILED
        self.state_map[step_id]["error"] = error

    def mark_skipped(self, step_id: str):
        """Mark a step as intentionally skipped."""
        self.state_map[step_id]["state"] = StepState.SKIPPED

    def is_finished(self) -> bool:
        """Return True if all steps are completed, skipped, or failed."""
        return all(
            self.state_map[s]["state"]
            in {StepState.COMPLETED, StepState.SKIPPED, StepState.FAILED}
            for s in self.state_map
        )

    # In agent/tool_chain_fsm.py, inside the ToolChainFSM class

    def get_step_output(self, step_id: str) -> Optional[Any]:
        """
        Safely retrieve the stored result for a completed step.
        """
        if step_id in self.state_map and self.state_map[step_id]["state"] == StepState.COMPLETED:
            return self.state_map[step_id].get("result")
        return None

    def get_plan_for_step(self, step_id: str) -> Dict[str, Any]:
        """
        Return the tool call dict for a given step ID.

        Args:
            step_id (str): ID in the form 'step_n'.

        Returns:
            dict: { "tool": str, "params": dict } for the given step.
        """
        step_index = int(step_id.split("_")[1])
        step = self.plan.steps[step_index]
        return {"tool": step.tool, "params": step.params}
