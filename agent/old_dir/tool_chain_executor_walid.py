"""agent/tool_chain_executor.py"""
import os
import logging
import re
import json
from typing import Any, Dict, List, Optional
from agent.tool_chain_fsm import ToolChainFSM
from agent_models.agent_models import ToolChain
from agent_models.step_state import StepState
from agent_models.step_status import StepStatus
from tools.tool_models import ToolResult, ToolResults
from tools.tool_registry import ToolRegistry
from agent.planner import Planner

logger = logging.getLogger(__name__)

# Matches <step_0> or <step_0.result> or <step_0.result.contents>
_PLACEHOLDER_RE = re.compile(r"^<step_(\d+)(?:\.(.+))?>$")            # exact match
_PLACEHOLDER_EMBEDDED_RE = re.compile(r"<step_(\d+)(?:\.(.+?))?>")    # embedded


class ToolChainExecutor:
    def __init__(
        self,
        plan: ToolChain = None,
        planner: Planner = None,
        strict_dependency_resolution: bool | None = None,
    ):
        self.registry = ToolRegistry()
        self.plan = plan
        self.planner = planner
        if strict_dependency_resolution is None:
            env_flag = os.getenv("AGENT_STRICT_DEPS", "").strip().lower()
            self.strict_dependency_resolution = env_flag in {"1", "true", "yes", "on"}
        else:
            self.strict_dependency_resolution = bool(strict_dependency_resolution)
        logger.info("[Executor] Strict dependency resolution: %s",
                    "ON" if self.strict_dependency_resolution else "OFF")

    def run_plan_with_fsm(self, plan: ToolChain = None) -> ToolResults:
        plan = plan or self.plan
        if not plan:
            logger.error("[Executor] No plan provided.")
            return ToolResults(results=[])
        logger.info(f"[Executor] Running plan with {len(plan.steps)} steps")
        fsm = ToolChainFSM(plan)
        executed_results: List[ToolResult] = []
        while not fsm.is_finished():
            step_id = fsm.get_next_pending_step()
            if not step_id:
                logger.debug("[Executor] No runnable step found, terminating.")
                break
            step_data = fsm.get_plan_for_step(step_id)
            tool_name = step_data["tool"]
            try:
                params = self._resolve_step_dependencies(step_data["params"], fsm, step_id)
            except Exception as e:
                error_message = f"Dependency resolution failed: {e}"
                logger.error(f"Failed to resolve dependencies for {step_id}: {e}", exc_info=True)
                fsm.mark_failed(step_id, error_message)
                executed_results.append(ToolResult(
                    step_id=step_id, tool=tool_name, params=step_data["params"],
                    status=StepStatus.ERROR, state=StepState.FAILED, message=error_message
                ))
                continue

            logger.info(f"[Executor] Running {step_id}: tool='{tool_name}' params={params}")
            result_entry = ToolResult(step_id=step_id, tool=tool_name, params=params, state=StepState.IN_PROGRESS)
            fsm.mark_in_progress(step_id)
            try:
                output = self._run_tool_once(tool_name, params)
                status = self._coerce_status(output.get("status", StepStatus.SUCCESS))
                message = output.get("message", "")
                result = output.get("result")
                fsm.mark_completed(step_id, result)
                result_entry.status = status
                result_entry.message = message
                result_entry.result = result
                result_entry.state = StepState.COMPLETED
            except Exception as e:
                error_message = str(e)
                logger.error(f"[Executor] Step {step_id} failed: {e}", exc_info=True)
                fsm.mark_failed(step_id, error_message)
                result_entry.status = StepStatus.ERROR
                result_entry.message = error_message
                result_entry.state = StepState.FAILED
            executed_results.append(result_entry)
        logger.info(f"[Executor] Plan execution complete, {len(executed_results)} results collected.")
        return ToolResults(results=executed_results)

    def _coerce_status(self, status: Any) -> StepStatus:
        if isinstance(status, str):
            return StepStatus(status.lower())
        return status if isinstance(status, StepStatus) else StepStatus.SUCCESS

    # -------------------- Placeholder resolution (type‑preserving) --------------------

    def _get_completed_index(self, step_id: str) -> Optional[int]:
        try:
            return int(step_id.split("_")[1])
        except Exception:
            return None

    def _raise_or_empty(self, msg: str) -> str:
        if self.strict_dependency_resolution:
            raise ValueError(msg)
        logger.warning(msg + " Using empty string.")
        return ""

    def _unwrap_result_envelope(self, value: Any) -> Any:
        """
        Normalize common envelope forms to the inner 'result' value.
        Handles:
          - objects with a .result attr
          - dicts like {'status': ..., 'message': ..., 'result': ...}
        """
        # object with .result
        if hasattr(value, "result"):
            try:
                return getattr(value, "result")
            except Exception:
                pass
        # dict envelope
        if isinstance(value, dict) and "result" in value and "status" in value:
            try:
                return value.get("result")
            except Exception:
                pass
        return value

    def _fetch_step_value(self, fsm: ToolChainFSM, ref_idx: int, current_idx: int,
                          path: List[str] | None, token: str) -> Any:
        if ref_idx >= current_idx:
            return self._raise_or_empty(
                f"Forward reference {token} to future step_{ref_idx} from step_{current_idx}."
            )
        ref_id = f"step_{ref_idx}"
        prev_output = fsm.get_step_output(ref_id)
        if prev_output is None:
            return self._raise_or_empty(
                f"No output available for referenced step: {ref_id} while resolving {token}."
            )

        value: Any = self._unwrap_result_envelope(prev_output)

        try:
            for attr in (path or []):
                if isinstance(value, dict):
                    value = value.get(attr, None)
                else:
                    value = getattr(value, attr, None)
                if value is None:
                    return self._raise_or_empty(
                        f"Attribute '{attr}' resolved to None in path {token}."
                    )
            value = self._unwrap_result_envelope(value)
            return value
        except Exception as e:
            return self._raise_or_empty(
                f"Failed to resolve path {token}: {e.__class__.__name__}: {e}"
            )

    def _resolve_step_dependencies(self, params: Any, fsm: ToolChainFSM, current_step_id: str) -> Any:
        """
        Recursively resolve placeholders in params (str, dict, list).
        - Exact placeholder string: return the underlying object (list/dict/etc) as-is (after unwrapping).
        - Embedded placeholder in a longer string: replace with JSON for lists/dicts, else str(value).
        - If an exact placeholder resolves to a list inside a list, splice it (extend) to avoid nested lists.
        """
        current_idx = self._get_completed_index(current_step_id) or 0

        if isinstance(params, dict):
            return {k: self._resolve_step_dependencies(v, fsm, current_step_id) for k, v in params.items()}

        if isinstance(params, list):
            out: List[Any] = []
            for v in params:
                if isinstance(v, str):
                    sv = v.strip()
                    m = _PLACEHOLDER_RE.match(sv)
                    if m:
                        ref_idx = int(m.group(1))
                        path = m.group(2).split(".") if m.group(2) else None
                        resolved = self._fetch_step_value(fsm, ref_idx, current_idx, path, m.group(0))
                        # unwrap once more in case a dict envelope slipped through
                        resolved = self._unwrap_result_envelope(resolved)
                        if isinstance(resolved, list):
                            out.extend(resolved)   # splice lists
                        else:
                            out.append(resolved)
                        continue
                out.append(self._resolve_step_dependencies(v, fsm, current_step_id))
            return out

        if isinstance(params, str):
            s = params.strip()

            m = _PLACEHOLDER_RE.match(s)
            if m:
                ref_idx = int(m.group(1))
                path = m.group(2).split(".") if m.group(2) else None
                resolved = self._fetch_step_value(fsm, ref_idx, current_idx, path, m.group(0))
                return self._unwrap_result_envelope(resolved)

            def _repl(m2: re.Match) -> str:
                ref_idx = int(m2.group(1))
                path = m2.group(2).split(".") if m2.group(2) else None
                val = self._fetch_step_value(fsm, ref_idx, current_idx, path, m2.group(0))
                val = self._unwrap_result_envelope(val)
                if isinstance(val, (list, dict)):
                    try:
                        return json.dumps(val)
                    except Exception:
                        return str(val)
                return "" if val is None else str(val)

            return _PLACEHOLDER_EMBEDDED_RE.sub(_repl, s)

        return params

    def _run_tool_once(self, tool_name: str, params: dict) -> Dict[str, Any]:
        if tool_name == "llm_response_async":
            tool_fn = self.registry.get_function(tool_name, planner=self.planner)
        else:
            tool_fn = self.registry.get_function(tool_name)
        output = self._normalize_output(tool_fn(**params))
        return output

    def _normalize_output(self, out: Any) -> Dict[str, Any]:
        if not isinstance(out, dict):
            out = {"status": StepStatus.SUCCESS, "result": out, "message": ""}
        if "status" not in out:
            out["status"] = StepStatus.SUCCESS
        else:
            out["status"] = self._coerce_status(out["status"])
        if "message" not in out:
            out["message"] = ""
        return out

