"""agent/tool_steps_executor.py"""

import logging
import re
import copy
from typing import Any, Dict, List
from tools.tool_registry import ToolRegistry
from agent.tool_chain_fsm import ToolChainFSM, StepState
from agent_models.llm_response_models import ToolResult, ToolResults, ToolChain

logger = logging.getLogger(__name__)


class ToolChainExecutor:
    """
    Executor for running a validated multi-step tool plan under FSM control.

    This class coordinates tool execution in sequence using a ToolChainFSM
    to enforce progression rules and track per-step states. It is designed
    to work with a preprocessed ToolChain plan where all inter-step references
    have already been resolved (e.g., "step_0" → "<step_0.result.files[0]>").

    Core Responsibilities:
        - Initialize an FSM to manage execution order and track states.
        - Iterate through plan steps, executing tools in sequence.
        - Enforce that a step only runs when its dependencies are marked
          as completed in the FSM.
        - Normalize tool outputs to a standard format:
              {"status": str, "result": Any, "message": str}
        - Record detailed execution results in ToolResult objects.
        - Update FSM state to COMPLETED, FAILED, or SKIPPED based on outcome.
        - Maintain an in-memory context of all tool outputs for later reference.
        - Optionally maintain a conversation-style chat history for LLMs.

    Execution Flow:
        1. FSM is initialized with all steps set to PENDING.
        2. Loop until FSM reports all steps are finished.
        3. Retrieve the next runnable step (PENDING + dependencies met).
        4. Mark it IN_PROGRESS, execute the corresponding tool from the registry.
        5. Normalize output, update FSM state to COMPLETED or FAILED.
        6. Append execution details to results list.
        7. Repeat until all steps are processed or execution is aborted.

    Notes:
        - All plan parameters should be pre-resolved; this executor does not
          perform placeholder resolution unless `_resolve_references()` is
          explicitly used.
        - Error handling is per-step; failures are logged in ToolResult and
          do not raise exceptions to the caller.
        - The class currently assumes linear dependency (step N depends on
          step N-1). Branching and parallel execution are not yet supported.

    Args:
        plan (List[Dict[str, Any]]):
            Ordered list of tool calls from the Planner, where each dict
            contains:
                "tool": str — registered tool name
                "params": dict — pre-resolved arguments for the tool

    Returns:
        ToolResults:
            A validated container of ToolResult objects with execution details
            for every step in the plan.

    Example:
        >>> executor = ToolChainExecutor(plan)
        >>> results = executor.run_plan_with_fsm(plan)
        >>> for step in results.results:
        ...     print(step.step_id, step.state, step.status)
    """

    def __init__(self, plan: ToolChain):
        """
        Args:
            plan: list of tool call dicts, from Planner
        """
        self.registry = ToolRegistry()
        self.plan = plan
        self.state_map: Dict[str, Dict[str, Any]] = {}
        self._initialize_states()
        self.context: Dict[str, Dict[str, Any]] = {}  # step_n -> tool output
        self.chat_history: List[Dict[str, Any]] = (
            []
        )  # todo: optional LLM conversation log placeholder; use later!

    def _initialize_states(self):
        """Initialize all steps as pending"""
        for i, _ in enumerate(self.plan):
            self.state_map[f"step_{i}"] = {"state": StepState.PENDING, "result": None}

    def run_plan_with_fsm(self, plan: ToolChain) -> ToolResults:
        """
        Execute a multi-step tool plan using an FSM controller for progression.

        This method runs each step in the provided plan sequentially, with
        execution order and readiness controlled by a ToolChainFSM instance.
        The FSM enforces that a step is only executed when its dependencies
        (previous steps in a linear plan) are marked as completed.

        For each step:
            - Marks the step as in_progress in the FSM.
            - Executes the corresponding tool function from the registry.
            - Normalizes the tool's output to a standard {status, result, message}
                format.
            - Updates FSM state to completed or failed based on execution outcome.
            - Records execution details in a ToolResult instance.

        Args:
            plan (List[Dict[str, Any]]):
                Ordered list of tool call dictionaries, each with "tool" and "params".
                The params should be pre-resolved (no placeholder parsing done here).

        Returns:
            ToolResults:
                A validated container of ToolResult objects, each representing the
                execution outcome of one step in the plan. Includes the step ID,
                tool name, parameters, FSM state, execution status, message,
                and result.

        Raises:
            Any exception raised by tool execution will be caught and logged as a
            failed ToolResult; the exception is not re-raised.
        """
        # Initialize FSM & setup results holder
        fsm = ToolChainFSM(plan)
        executed_results: List[ToolResult] = []

        # * Main loop – keep running until FSM says all steps are done
        while not fsm.is_finished():
            step_id = fsm.get_next_pending_step()
            if not step_id:
                break  # no runnable step

            # ✅ Retrieve step details
            step_data = fsm.get_plan_for_step(step_id)
            tool_name = step_data["tool"]
            params = step_data["params"]  # should already resolved before execution
            # params = self._resolve_references(step_data["params"])

            # ✅ Create base result entry/obj (pending -> in_progress)
            result_entry = ToolResult(
                step_id=step_id,
                tool=tool_name,
                params=params,
                status="pending",
                state=StepState.IN_PROGRESS,
            )
            fsm.mark_in_progress(step_id)

            try:
                # ✅ Run the tool
                tool_fn = self.registry.get_function(tool_name)
                output = tool_fn(**params)

                # Normalize tool output
                if not isinstance(output, dict):
                    output = {"status": "ok", "result": output, "message": ""}

                # * ✅ Update result entry and FSM state
                result_entry.status = output.get("status", "success")
                result_entry.message = output.get("message", "")
                result_entry.result = output.get("result", output)
                result_entry.state = (
                    StepState.COMPLETED
                    if result_entry.status == "success"
                    else StepState.FAILED
                )

                self.context[step_id] = (
                    output  # Stores raw output in context for later reference resolution.
                )

                # ✅ Mark completion or failure
                if result_entry.state == StepState.COMPLETED:
                    fsm.mark_completed(step_id, output)
                else:
                    fsm.mark_failed(
                        step_id, result_entry.message or ""
                    )  # coerce to str

            except Exception as e:
                error_msg = str(e)
                result_entry.status = "error"
                result_entry.message = error_msg
                result_entry.state = StepState.FAILED
                fsm.mark_failed(step_id, error_msg)

            executed_results.append(result_entry)
        return ToolResults(results=executed_results)  # Return as pydantic model

    def _extract_path(self, source: Any, path: str) -> Any:
        """
        Traverse a dict/list using dot notation and [index] syntax.
        Example: result.files[0] -> source['result']['files'][0]
        """
        parts = re.split(r"\.(?![^\[]*\])", path)  # split on '.' not inside []
        val = source
        for part in parts:
            if "[" in part and part.endswith("]"):
                field, idx = part[:-1].split("[")
                if field:
                    val = val.get(field) if isinstance(val, dict) else None
                val = val[int(idx)] if isinstance(val, (list, tuple)) else None
            else:
                val = val.get(part) if isinstance(val, dict) else None
            if val is None:
                break
        return val

    def _log_chat_history_entry(self, step_id: str, entry: Dict[str, Any]):
        """
        Append a conversation-style log entry for LLM-friendly history.
        """
        self.chat_history.append({"role": "tool", "name": step_id, "content": entry})
