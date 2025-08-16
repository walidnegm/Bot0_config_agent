"""agent/tool_chain_executor.py"""

import os
import logging
import re
from typing import Any, Dict, List
from pydantic import BaseModel
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
from tools.workbench.tool_models import (
    ToolOutput,
    ToolResult,
    ToolResults,
)
from tools.workbench.tool_registry import ToolRegistry


logger = logging.getLogger(__name__)

EMPTY = (None, "", [], {}, ())
_PLACEHOLDER_RE = re.compile(r"^<([^>]+)>$")  # matches whole-string "< ... >"


class ToolChainExecutor:
    """
    Executor for running a validated multi-step tool plan under FSM control.

    This class coordinates tool execution in sequence using a ToolChainFSM
    to enforce progression rules and track per-step states.

    Core Responsibilities:
        - Initialize an FSM to manage execution order and track states.
        - Iterate through plan steps, executing tools in sequence.
        - Enforce that a step only runs when its dependencies are completed.
        - Normalize tool outputs to:
              {"status": StepStatus, "result": Any, "message": str}
        - Record detailed execution results in ToolResult objects.
        - Update FSM state to COMPLETED or FAILED based on outcome.
        - Maintain an in-memory context of all tool outputs (as ToolOutput models).
        - Support structured, model-validated handoff of params to the next step.

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
        ToolResults: Validated container of ToolResult objects with execution
            details.

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
        self.context: Dict[str, ToolOutput] = {}
        self.chat_history: List[Dict[str, Any]] = []  # LLM-style chat log (optional)

        # Scope (set by set_scope tool)
        self.project_root_dir: str | None = None
        self.branches: List[str] = []

        # Holder of the next step's auto-generated parameters
        self._next_params: dict[int, BaseModel] = {}

    # * Core function to execute the plan
    def run_plan_with_fsm(self, plan: ToolChain) -> ToolResults:
        """
        Execute a multi-step tool plan using an FSM controller for progression.

        This method runs each step in sequence under the control of a ToolChainFSM.
        Each step can inject structured parameters into the *next* step via
        `_next_params`, populated by automatic output→input conversion in the
        ToolRegistry. The executor records normalized results in `context` for
        debugging and reference.

        Workflow
        --------
        ToolChain → FSM (get runnable step)
        → inject `_next_params` if present
        → execute tool → normalize output (ToolOutput)
        → registry.match_and_convert_output(...) for next tool
        → store validated next-input model in `_next_params`
        → update FSM + record ToolResult
        → repeat
        --------

            +---------------------+
            |  ToolChain (plan)   |
            +----------+----------+
                       |
                       v
            +----------+----------+            +-----------------------+
            |   ToolChainFSM      |            |   ToolRegistry        |
            |  (state: PENDING→*) |            |  (tools + transforms) |
            +----------+----------+            +-----------+-----------+
                       |                                      ^
                       | get_next_pending_step()              |
                       v                                      |
            +----------+----------+                          |
            |  Inject _next_params |                          |
            |  (if available)      |--------------------------+
            +----------+----------+
                       |
                       v
            +----------+----------+
            |   Execute tool      |  → tool_fn(**params)
            +----------+----------+
                       |
                       v
            +----------+----------+
            | Normalize output    |  → {"status", "result", "message"}
            +----------+----------+
                       |
                       |  If next step exists:
                       |    next_tool = plan.steps[i+1].tool
                       |    next_input = registry.match_and_convert_output(
                       |        output=ToolOutput(**output),
                       |        target_tool=next_tool
                       |    )
                       |  (Self-validate or transform, then store in _next_params)
                       v
            +----------+----------+
            | Update FSM & store  |  context["step_i"] = normalized output
            | result (ToolResult) |
            +----------+----------+
                       |
                       v
                Loop until FSM.is_finished()

        Input/Output Matching & Conversion
        ----------------------------------
        - After executing a step, the executor may attempt to convert its normalized
          output into the input model of the *next* tool:
            registry.match_and_convert_output(output, target_tool)
        - If a transform mapping exists for (source_tool, target_tool) in
          `tool_transformation.json`:
            - `transform_fn: null` → **self-validation**: feed output directly into
              the next tool’s Pydantic input model.
            - `transform_fn` set → import and apply that function before validation.
        - On successful conversion, the resulting dict is stored in `_next_params[next_idx]`.
          On failure, the executor logs a warning and continues without injection.

        Step Execution Flow
        -------------------
        - Marks the step IN_PROGRESS in the FSM.
        - Merges auto-injected params from `_next_params` with the plan’s own params
          (plan params override injected ones).
        - Executes the tool function from the registry.
        - Normalizes output to {"status", "result", "message"}.
        - Optionally performs output→input conversion for the next step.
        - Updates FSM to COMPLETED or FAILED.
        - Records a ToolResult and stores the output in execution context.

        Args
        ----
        plan : ToolChain
            A validated ToolChain with resolved parameters.

        Returns
        -------
        ToolResults
            One ToolResult per step with step ID, tool, params, FSM state,
            status, message, and result.

        Notes
        -----
        All exceptions during execution are caught and stored in the corresponding
        ToolResult; the executor never re-raises, ensuring the FSM can track failure
        states without aborting the whole plan.
        """

        # Use the plan held by the instance (copy already constructed elsewhere)
        plan = self.plan

        logger.info("[Executor] Running plan with %d steps", len(plan.steps))
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
            tool_name: str = step_data["tool"]
            params: dict = step_data["params"]

            curr_idx = int(step_id.split("_")[1])

            # ---- Create a copy of params ----
            params = dict(
                params or {}
            )  # --> used for execution (keeps Path, Enums, etc.)
            params_json = dict(params or {})  # --> used for logs/results (JSON-safe)

            # Debug: show initial planner params
            logger.debug("[Executor] %s initial plan params: %s", step_id, params)

            # ---- Inject validated defaults for this step (if available) ----
            if curr_idx in self._next_params:
                auto_model = self._next_params.pop(
                    curr_idx
                )  # validated BaseModel or None (defensive)
                if auto_model is None:
                    logger.debug(
                        "[Executor] %s has empty _next_params; skipping injection.",
                        step_id,
                    )
                else:
                    injected = auto_model.model_dump()  # for execution (keeps Path)
                    injected_json = auto_model.model_dump(
                        mode="json"
                    )  # for logging/ToolResult.params

                    logger.debug(
                        "[Executor] Injecting validated params from %s",
                        auto_model.__class__.__name__,
                    )
                    logger.debug(
                        "[Executor] %s found _next_params for injection: %s",
                        step_id,
                        injected_json,
                    )
                    logger.debug(
                        "[Executor] %s planner params before merge: %s",
                        step_id,
                        (params or {}),
                    )

                    # Prefer injected (validated); fall back to planner where injected not meaningful
                    params = self._resolve_btw_step_params(
                        injected=injected,
                        planner=(params or {}),
                        step_id=step_id,
                        tool_name=tool_name,
                    )
                    params_json = self._resolve_btw_step_params(
                        injected=injected_json,
                        planner=(params or {}),
                        step_id=step_id,
                        tool_name=tool_name,
                    )

                    logger.debug("[%s] params (exec): %s", step_id, params)
                    logger.debug("[%s] params (log) : %s", step_id, params_json)

            # ---- Execute the tool ----
            logger.info(
                "[Executor] Running %s: tool='%s' params=%s", step_id, tool_name, params
            )
            result_entry = ToolResult(
                step_id=step_id,
                tool=tool_name,
                params=params_json,  # << JSON-safe for output & logging
                status=None,
                state=StepState.IN_PROGRESS,
            )
            fsm.mark_in_progress(step_id)

            try:
                # Scope tool handled like a normal tool, but we also update executor state.
                if tool_name == "set_scope":
                    tool_fn = self.registry.get_function(tool_name)
                    output = self._normalize_output(tool_fn(**params))

                    payload = (
                        output.get("result", {})
                        if isinstance(output.get("result"), dict)
                        else {}
                    )
                    self.project_root_dir = (
                        payload.get("dir") or params.get("dir") or self.project_root_dir
                    )
                    self.branches = (
                        payload.get("branches") or params.get("branches") or []
                    )
                    if self.branches is None:
                        self.branches = []

                else:
                    # Inject default (root) dir when useful
                    params = self._inject_default_root_dir(tool_name, params)

                    # Fan-out over branches if supported
                    if self.branches and self._tool_accepts_root_dir(tool_name):
                        output = self._fanout_over_branches(tool_name, params)
                    else:
                        output = self._run_tool_once(tool_name, params)

                # ---- Normalize output envelope (single place) ----
                if not isinstance(output, dict):
                    output = {
                        "status": StepStatus.SUCCESS,
                        "result": output,
                        "message": "",
                    }

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

                # Store context for traceability
                tool_output_model = ToolOutput(**output)
                self.context[step_id] = tool_output_model

                # Log normalized envelope
                logger.debug(
                    "[Executor] %s normalized output: status=%s, message=%s, result_keys=%s",
                    step_id,
                    output.get("status"),
                    output.get("message"),
                    (
                        list(output.get("result", {}).keys())
                        if isinstance(output.get("result"), dict)
                        else type(output.get("result"))
                    ),
                )

                # ---- Convert for NEXT step only if this step succeeded ----
                if status != StepStatus.SUCCESS:
                    logger.debug(
                        "[Executor] %s %s: skipping next-step conversion due to non-success status=%s",
                        step_id,
                        tool_name,
                        status,
                    )
                else:
                    try:
                        next_idx = curr_idx + 1
                        if 0 <= next_idx < len(plan.steps):
                            next_tool = plan.steps[next_idx].tool
                            logger.debug(
                                "[Executor] %s attempting convert for step_%d (%s) from '%s'",
                                step_id,
                                next_idx,
                                next_tool,
                                tool_name,
                            )

                            next_input_model = self.registry.match_and_convert_output(
                                output=tool_output_model,
                                target_tool=next_tool,
                                source_tool=tool_name,
                            )

                            if next_input_model is not None:
                                self._next_params[next_idx] = next_input_model
                                logger.debug(
                                    "[Executor] %s stored _next_params[%d] = %s",
                                    step_id,
                                    next_idx,
                                    next_input_model.model_dump(),
                                )
                            else:
                                logger.debug(
                                    "[Executor] %s no _next_params for step_%d (convert returned None)",
                                    step_id,
                                    next_idx,
                                )
                    except Exception as e:
                        logger.warning(
                            "[Executor] %s failed next-step conversion %s → %s: %s",
                            step_id,
                            tool_name,
                            (
                                plan.steps[next_idx].tool
                                if 0 <= next_idx < len(plan.steps)
                                else "N/A"
                            ),
                            e,
                        )

                # ---- Final result log + FSM update ----
                logger.debug(
                    "[Executor] Step %s: tool='%s', status=%s, state=%s, message=%s",
                    step_id,
                    tool_name,
                    result_entry.status,
                    result_entry.state,
                    result_entry.message,
                )
                if result_entry.state == StepState.COMPLETED:
                    fsm.mark_completed(step_id, output)
                else:
                    fsm.mark_failed(step_id, result_entry.message or "")

            except Exception as e:
                # Any unhandled exception marks the step failed; keep FSM moving.
                error_msg = str(e)
                logger.error(
                    "[Executor] Step %s failed: %s", step_id, error_msg, exc_info=True
                )
                result_entry.status = StepStatus.ERROR
                result_entry.message = error_msg
                result_entry.state = StepState.FAILED
                fsm.mark_failed(step_id, error_msg)

            executed_results.append(result_entry)

        logger.info(
            "[Executor] Plan execution complete, %d results collected.",
            len(executed_results),
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
        # Validate/allow tool-specific extras via FanoutParams (dir is optional)
        base = FanoutParams.model_validate(params)

        merged = FanoutResult()  # default: status=SUCCESS, message="", per_branch=[]
        msgs: list[str] = []

        for rel in self.branches:
            # Build branch-specific params (copy + injected (root) dir))
            bparams = base.model_dump()  # dict with extra fields preserved
            bparams["dir"] = (
                os.path.join(self.project_root_dir or "", rel)
                if self.project_root_dir
                else rel
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
            payload_model.per_branch.append(branch_rec)  # pylint: disable=no-member

            if branch_rec.status != StepStatus.SUCCESS:
                merged.status = StepStatus.ERROR
                if branch_rec.message:
                    msgs.append(f"{rel}: {branch_rec.message}")

        if msgs:
            merged.message = "; ".join(msgs)

        # Store typed, but executor/context expects a dict envelope — dump it
        return merged.model_dump()

    def _inject_default_root_dir(self, tool_name: str, params: dict) -> Dict[str, Any]:
        if not self.project_root_dir or not self._tool_accepts_root_dir(tool_name):
            return params
        dir = params.get("dir")
        if dir in (None, "", ".", "./"):
            params = dict(params)
            params["dir"] = self.project_root_dir
        return params

    def _initialize_states(self):
        """Initialize all steps as pending in the state map."""
        for i, _ in enumerate(self.plan.steps):
            self.state_map[f"step_{i}"] = {"state": StepState.PENDING, "result": None}

    def _is_meaningful_param(self, v: Any) -> bool:
        """
        # todo: should be expanded later to include more validation logics.

        Check whether a parameter value is considered meaningful for tool execution.

        A value is considered non-meaningful if it is:
            - None
            - Present in the EMPTY sentinel set
            - An empty string after stripping whitespace

        Args:
            v (Any): The parameter value to check.

        Returns:
            bool: True if the value is non-empty and usable, False otherwise.
        """
        if v in EMPTY:
            return False
        if isinstance(v, str) and v.strip() == "":
            return False
        return True

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

    def _resolve_btw_step_params(
        self,
        injected: dict | None,
        planner: dict | None,
        *,
        step_id: str,
        tool_name: str,
    ) -> dict:
        """
        Resolve the effective parameters for a tool step by merging injected (validated)
        and planner-provided values.

        Priority:
            - Prefer injected parameters if they are considered meaningful.
            - Fall back to planner parameters if injected values are missing or not meaningful.
            - Leave values empty if neither source provides meaningful data.

        Diagnostics:
            - Logs which keys were taken from injected vs. planner.
            - Issues warnings if critical keys (e.g., "files", "path", "paths", "dir")
            remain unfilled after merging.

        Args:
            injected (dict | None): Validated parameters (usually from transformation or previous step).
            planner (dict | None): Parameters originally proposed by the planner (LLM).
            step_id (str): Identifier of the step for logging.
            tool_name (str): Name of the tool for logging.

        Returns:
            dict: The merged parameter dictionary, with injected values prioritized.
        """
        injected = dict(injected or {})
        planner = dict(planner or {})

        merged = {}
        keys = set(injected) | set(planner)

        took_from_injected, took_from_planner = [], []

        for k in keys:
            v_inj = injected.get(k, None)
            if self._is_meaningful_param(v_inj):
                merged[k] = v_inj
                took_from_injected.append(k)
            else:
                merged[k] = planner.get(k, None)
                if k in planner:
                    took_from_planner.append(k)

        # Diagnostics
        if took_from_injected:
            logger.debug(
                "[Executor] %s %s params: took from injected → %s",
                step_id,
                tool_name,
                took_from_injected,
            )
        if took_from_planner:
            logger.debug(
                "[Executor] %s %s params: fell back to planner → %s",
                step_id,
                tool_name,
                took_from_planner,
            )

        # Warn if critical keys remain unfilled
        critical = ("files", "path", "paths", "dir")
        crit_missing = [
            k
            for k in critical
            if k in keys
            and not self._is_meaningful_param(injected.get(k))
            and not self._is_meaningful_param(planner.get(k))
        ]
        if crit_missing:
            logger.warning(
                "[Executor] %s %s params: critical keys still unfilled → %s",
                step_id,
                tool_name,
                crit_missing,
            )

        return merged

    def _resolve_planner_placeholders(self, params: Any, context: dict) -> Any:
        """
        Recursively resolve planner placeholders that reference prior steps, e.g.:
            "<step_0.result.files>" or "<step_2.result.files[0]>"
        Notes:
            - '<prev_output>' is intentionally unsupported/removed.
            - Non-matching strings are returned unchanged.
            - If a token can't be resolved, returns None for that leaf.
        """
        if isinstance(params, dict):
            return {
                k: self._resolve_planner_placeholders(v, context)
                for k, v in params.items()
            }
        if isinstance(params, list):
            return [self._resolve_planner_placeholders(v, context) for v in params]
        if isinstance(params, str):
            m = _PLACEHOLDER_RE.match(params)
            if not m:
                return params  # plain string
            token = m.group(1)  # e.g. "step_1.result.files[0]"
            return self._resolve_token(token, context)
        return params

    # TODO: for future scope expansion / do not delete
    def _resolve_token(self, token: str, context: dict) -> Any:
        """
        Split a single token like 'step_1.result.files[0]'.

        Returns None if it can't be resolved.

        >>> Example:
            "step_1.result.files[0]"
            → step_id = "step_1"
            → sub_path = "result.files[0]"
            → look up context["step_1"], then walk down .result → .files → [0]
        """
        # Only support explicit step references
        if not token.startswith("step_"):
            # legacy/deprecated tokens (e.g., prev_output) intentionally not supported
            return None

        # Split "step_1.result.files[0]" -> step_id="step_1", sub_path="result.files[0]" (optional)
        if "." in token:
            step_id, sub_path = token.split(".", 1)
        else:
            step_id, sub_path = token, None

        step_obj = context.get(step_id)
        if step_obj is None:
            return None

        # Unwrap ToolOutput-like objects to dict for path extraction
        base = None
        if hasattr(step_obj, "model_dump"):
            # Pydantic model (e.g., ToolOutput)
            step_dict = step_obj.model_dump()
            base = step_dict
        elif hasattr(step_obj, "result"):
            base = {"result": step_obj.result}
        elif isinstance(step_obj, dict):
            base = step_obj
        else:
            # Unknown shape; fall back to raw
            base = {"value": step_obj}

        if sub_path:
            # First try path as given; if not found and we have a 'result' key, try within result.
            val = self._extract_path(base, sub_path)
            if val is None and isinstance(base, dict) and "result" in base:
                val = self._extract_path(base["result"], sub_path)
            return val
        else:
            # No subpath: prefer the 'result' payload if present
            if isinstance(base, dict) and "result" in base:
                return base["result"]
            return base

    def _run_tool_once(self, tool_name: str, params: dict) -> Dict[str, Any]:  # NEW
        tool_fn = self.registry.get_function(tool_name)
        output = self._normalize_output(tool_fn(**params))
        return output

    def _tool_accepts_root_dir(self, tool_name: str) -> bool:
        """
        Return:
            True if "dir" is one of the valid parameter names in that tool’s
                JSON schema.
            False if "dir" is not there or if the lookup/validation fails
                (your except Exception).
        """
        try:
            return "dir" in self.registry.get_param_keys(tool_name)
        except Exception:
            return False
