"""agent/tool_chain_executor.py"""

import os
import logging
import re
from typing import Any, Dict, List
from agent.tool_chain_fsm import ToolChainFSM
from agent_models.agent_models import ToolChain
from agent_models.step_state import StepState
from agent_models.step_status import StepStatus
from agent_models.fanout_models import (
    FanoutParams,
    BranchOutput,
    FanoutResult,
    FanoutPayload,
)
from tools.tool_models import (
    ToolResult,
    ToolResults,
)
from tools.tool_registry import ToolRegistry


logger = logging.getLogger(__name__)


class ToolChainExecutor:
    """
    Executor for running a validated multi-step tool plan under FSM control.

    This class coordinates tool execution in sequence using a ToolChainFSM
    to enforce progression rules and track per-step states. It resolves
    inter-step references like "<prev_output>" before tool execution.

    Core Responsibilities:
        - Initialize an FSM to manage execution order and track states.
        - Iterate through plan steps, executing tools in sequence.
        - Enforce that a step only runs when its dependencies are marked
          as completed in the FSM.
        - Normalize tool outputs to a standard format:
              {"status": StepStatus, "result": Any, "message": str}
        - Record detailed execution results in ToolResult objects.
        - Update FSM state to COMPLETED, FAILED, or SKIPPED based on outcome.
        - Maintain an in-memory context of all tool outputs for later reference.
        - Optionally maintain a conversation-style chat history for LLMs.

    Execution Flow:
        1. FSM is initialized with all steps set to PENDING.
        2. Loop until FSM reports all steps are finished.
        3. Retrieve the next runnable step (PENDING + dependencies met).
        4. Mark it IN_PROGRESS, resolve param placeholders, and execute the tool.
        5. Normalize output, update FSM state to COMPLETED or FAILED.
        6. Append execution details to results list.
        7. Repeat until all steps are processed or execution is aborted.

    Notes:
        - Error handling is per-step; failures are logged in ToolResult and
          do not raise exceptions to the caller.
        - The class currently assumes linear dependency (step N depends on
          step N-1). Branching and parallel execution are not yet supported.

    Args:
        plan (ToolChain): Validated ToolChain with resolved steps.

    Returns:
        ToolResults: Validated container of ToolResult objects with execution details.

    Example:
        >>> executor = ToolChainExecutor(plan)
        >>> results = executor.run_plan_with_fsm(plan)
        >>> for step in results.results:
        ...     print(step.step_id, step.state, step.status)
    """

    def __init__(self, plan: ToolChain):
        """
        Args:
            plan (ToolChain): Validated ToolChain plan to execute.
        """
        self.registry = ToolRegistry()
        self.plan = plan
        self.state_map: Dict[str, Dict[str, Any]] = {}
        self._initialize_states()
        self.context: Dict[str, Dict[str, Any]] = {}  # step_n -> output
        self.chat_history: List[Dict[str, Any]] = []  # LLM-style chat log (optional)

        # Scope (set by set_scope tool)
        self.project_root: str | None = None
        self.branches: List[str] = []

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
        logger.info(f"[Executor] Running plan with {len(plan.steps)} steps")
        fsm = ToolChainFSM(plan)
        executed_results: List[ToolResult] = []

        while not fsm.is_finished():
            step_id = fsm.get_next_pending_step()
            if not step_id:
                logger.debug(
                    "[Executor] No runnable step found, terminating execution loop."
                )
                break

            step_data = fsm.get_plan_for_step(step_id)
            tool_name = step_data["tool"]
            params = step_data["params"]

            # Resolve placeholders BEFORE tool call
            params = self._resolve_params(step_data["params"], self.context)
            logger.info(
                f"[Executor] Running {step_id}: tool='{tool_name}' params={params}"
            )
            result_entry = ToolResult(
                step_id=step_id,
                tool=tool_name,
                params=params,
                status=None,  # temp hardcoded holder
                state=StepState.IN_PROGRESS,
            )
            fsm.mark_in_progress(step_id)

            try:
                if tool_name == "set_scope":
                    # Treat set_scope like a normal tool call (it can validate/expand paths)
                    tool_fn = self.registry.get_function(tool_name)
                    output = self._normalize_output(tool_fn(**params))

                    # Capture scope (prefer tool output, fall back to params)
                    payload = (
                        output.get("result", {})
                        if isinstance(output.get("result"), dict)
                        else {}
                    )
                    self.project_root = (
                        payload.get("root") or params.get("root") or self.project_root
                    )
                    self.branches = (
                        payload.get("branches") or params.get("branches") or []
                    )
                    if self.branches is None:
                        self.branches = []

                else:
                    # Non_scope tools: inject default root when useful
                    params = self._inject_default_root(tool_name, params)

                    # Fan-out if we have branches and the tool accepts root
                    if self.branches and self._tool_accepts_root(tool_name):
                        output = self._fanout_over_branches(tool_name, params)
                    else:
                        output = self._run_tool_once(tool_name, params)

                # Normalize (common)
                if not isinstance(output, dict):
                    output = {
                        "status": StepStatus.SUCCESS,
                        "result": output,
                        "message": "",
                    }

                # Coerce enum class for status
                raw_status = output.get("status", StepStatus.SUCCESS)
                status = self._coerce_status(raw_status)

                result_entry.status = status
                result_entry.message = output.get("message", "")
                result_entry.result = output.get("result", output)
                result_entry.state = (
                    StepState.COMPLETED
                    if status == StepStatus.SUCCESS
                    else StepState.FAILED
                )

                self.context[step_id] = output

                # Log result for debugging
                logger.debug(
                    f"[Executor] Step {step_id}: tool='{tool_name}', status={result_entry.status}, "
                    f"state={result_entry.state}, message={result_entry.message}"
                )

                # Mark FSM state
                if result_entry.state == StepState.COMPLETED:
                    fsm.mark_completed(step_id, output)
                else:
                    fsm.mark_failed(step_id, result_entry.message or "")

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"[Executor] Step {step_id} failed: {error_msg}", exc_info=True
                )
                result_entry.status = StepStatus.ERROR
                result_entry.message = error_msg
                result_entry.state = StepState.FAILED
                fsm.mark_failed(step_id, error_msg)

            executed_results.append(result_entry)
        logger.info(
            f"[Executor] Plan execution complete, {len(executed_results)} results collected."
        )
        return ToolResults(results=executed_results)

    def _coerce_status(self, v) -> StepStatus:
        if isinstance(v, StepStatus):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "success":
                return StepStatus.SUCCESS
            if s == "error":
                return StepStatus.ERROR
        # default
        return StepStatus.SUCCESS

    def _extract_path(self, source: Any, path: str) -> Any:
        """
        Traverse a dict/list using dot notation and [index] syntax.
        Example: "result.files[0]" -> source['result']['files'][0]
        """
        if not path:
            return source
        parts = re.split(r"\.(?![^\[]*\])", path)  # Split on . not inside []
        val = source
        for part in parts:
            if "[" in part and part.endswith("]"):
                # e.g., files[3]
                field, idx = part[:-1].split("[")
                if field:
                    val = val.get(field) if isinstance(val, dict) else None
                val = val[int(idx)] if isinstance(val, (list, tuple)) else None
            else:
                val = val.get(part) if isinstance(val, dict) else None
            if val is None:
                break
        return val

    def _fanout_over_branches(self, tool_name: str, params: Dict) -> Dict:
        """
        A parallelizer/aggregator for running the same tool over multiple
        branch paths, collecting each branch’s result, and rolling them up
        into one unified status + message.

        Expected output:
            Dict form of FanoutResult (pydantic model).

        """
        # Validate/allow tool-specific extras via FanoutParams (root is optional)
        base = FanoutParams.model_validate(params)

        merged = FanoutResult()  # default: status=SUCCESS, message="", per_branch=[]
        msgs: list[str] = []

        for rel in self.branches:
            # Build branch-specific params (copy + injected root)
            bparams = base.model_dump()  # dict with extra fields preserved
            bparams["root"] = (
                os.path.join(self.project_root or "", rel) if self.project_root else rel
            )

            out = self._run_tool_once(tool_name, bparams)

            # Normalize tool output shape (dict w/ status, message, result)

            raw_status = out.get(
                "status", StepStatus.SUCCESS
            )  # Coerce enum class for status
            status = self._coerce_status(raw_status)
            message = out.get("message", "")
            payload = out.get("result", out)

            branch_rec = BranchOutput(
                branch=rel, status=status, message=message, output=payload
            )

            # Give a strongly type name (for pylint)
            payload_model: FanoutPayload = merged.result
            payload_model.per_branch.append(branch_rec)  # type: ignore[attr-defined]

            if branch_rec.status != StepStatus.SUCCESS:
                merged.status = StepStatus.ERROR
                if branch_rec.message:
                    msgs.append(f"{rel}: {branch_rec.message}")

        if msgs:
            merged.message = "; ".join(msgs)

        # Store typed, but executor/context expects a dict envelope — dump it
        return merged.model_dump()

    def _inject_default_root(self, tool_name: str, params: dict) -> Dict[str, Any]:
        if not self.project_root or not self._tool_accepts_root(tool_name):
            return params
        root = params.get("root")
        if root in (None, "", ".", "./"):
            params = dict(params)
            params["root"] = self.project_root
        return params

    def _initialize_states(self):
        """Initialize all steps as pending in the state map."""
        for i, _ in enumerate(self.plan.steps):
            self.state_map[f"step_{i}"] = {"state": StepState.PENDING, "result": None}

    def _log_chat_history_entry(self, step_id: str, entry: Dict[str, Any]):
        """
        Append a conversation-style log entry for LLM-friendly history.
        """
        self.chat_history.append({"role": "tool", "name": step_id, "content": entry})

    def _normalize_output(self, out: Any) -> Dict[str, Any]:
        """Helper to normalize any tool output and coerce status"""
        if not isinstance(out, dict):
            out = {"status": StepStatus.SUCCESS, "result": out, "message": ""}
        if "status" not in out:
            out["status"] = StepStatus.SUCCESS
        else:
            out["status"] = self._coerce_status(out["status"])
        if "message" not in out:
            out["message"] = ""
        return out

    def _resolve_reference(self, ref: str, context: dict) -> Any:
        """
        Resolves a reference string like '<prev_output.files[0]>' or '<step_1.result.files>'.
        """
        if not (ref.startswith("<") and ref.endswith(">")):
            return ref
        ref_path = ref[1:-1]
        if "." in ref_path:
            step_key, sub_path = ref_path.split(".", 1)
        else:
            step_key, sub_path = ref_path, None

        if step_key == "prev_output":
            steps = [k for k in context.keys() if k.startswith("step_")]
            if not steps:
                return None
            max_step = max(steps, key=lambda s: int(s.split("_")[1]))
            val = context[max_step]
        else:
            val = context.get(step_key)
        if val is None:
            return None
        if sub_path:
            return self._extract_path(val, sub_path)
        return val

    def _resolve_params(self, params: Any, context: dict) -> Any:
        """Recursively resolve all <...> references in params using context."""
        if isinstance(params, dict):
            return {k: self._resolve_params(v, context) for k, v in params.items()}
        elif isinstance(params, list):
            return [self._resolve_params(v, context) for v in params]
        elif (
            isinstance(params, str) and params.startswith("<") and params.endswith(">")
        ):
            return self._resolve_reference(params, context)
        else:
            return params

    def _run_tool_once(self, tool_name: str, params: dict) -> Dict[str, Any]:  # NEW
        tool_fn = self.registry.get_function(tool_name)
        output = self._normalize_output(tool_fn(**params))
        return output

    def _tool_accepts_root(self, tool_name: str) -> bool:
        spec = self.registry.tools.get(tool_name, {})
        props = spec.get("parameters", {}).get("properties", {})
        return "root" in props
