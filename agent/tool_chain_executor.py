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
from agent.planner import Planner

logger = logging.getLogger(__name__)

class ToolChainExecutor:
    """
    Executor for running a validated multi-step tool plan under FSM control.
    """
    def __init__(self, plan: ToolChain, planner: Planner = None):
        self.registry = ToolRegistry()
        self.plan = plan
        self.planner = planner

    def run_plan_with_fsm(self, plan: ToolChain) -> ToolResults:
        """
        Execute a multi-step tool plan using an FSM controller for progression.
        """
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
                # Resolve dependencies by passing the full FSM object
                params = self._resolve_step_dependencies(step_data["params"], fsm)
            except Exception as e:
                error_message = f"Dependency resolution failed: {e}"
                logger.error(f"Failed to resolve dependencies for {step_id}: {e}", exc_info=True)
                fsm.mark_failed(step_id, error_message)
                result_entry = ToolResult(
                    step_id=step_id, tool=tool_name, params=step_data["params"],
                    status=StepStatus.ERROR, state=StepState.FAILED,
                    message=error_message
                )
                executed_results.append(result_entry)
                continue

            logger.info(f"[Executor] Running {step_id}: tool='{tool_name}' params={params}")

            result_entry = ToolResult(
                step_id=step_id, tool=tool_name, params=params,
                status=None, state=StepState.IN_PROGRESS
            )
            fsm.mark_in_progress(step_id)

            try:
                output = self._run_tool_once(tool_name, params)
                status = self._coerce_status(output.get("status", StepStatus.SUCCESS))
                result_entry.status = status
                result_entry.message = output.get("message", "")
                result_entry.result = output.get("result", output)
                result_entry.state = StepState.COMPLETED if status == StepStatus.SUCCESS else StepState.FAILED

                logger.debug(f"[Executor] Step {step_id} finished with state: {result_entry.state}")

                if result_entry.state == StepState.COMPLETED:
                    fsm.mark_completed(step_id, output)
                else:
                    fsm.mark_failed(step_id, result_entry.message or "")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[Executor] Step {step_id} failed: {error_msg}", exc_info=True)
                result_entry.status = StepStatus.ERROR
                result_entry.message = error_msg
                result_entry.state = StepState.FAILED
                fsm.mark_failed(step_id, error_msg)

            executed_results.append(result_entry)

        logger.info(f"[Executor] Plan execution complete, {len(executed_results)} results collected.")
        return ToolResults(results=executed_results)

    def _coerce_status(self, v) -> StepStatus:
        if isinstance(v, StepStatus): return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "success": return StepStatus.SUCCESS
            if s == "error": return StepStatus.ERROR
        return StepStatus.SUCCESS
        
    def _resolve_step_dependencies(self, params: Any, fsm: ToolChainFSM) -> Any:
        """
        Recursively resolves parameter values that reference previous step outputs,
        including placeholders embedded within strings.
        """
        if isinstance(params, dict):
            return {k: self._resolve_step_dependencies(v, fsm) for k, v in params.items()}
        
        elif isinstance(params, list):
            return [self._resolve_step_dependencies(v, fsm) for v in params]

        elif isinstance(params, str):
            placeholder_pattern = r"<([^>]+)>"

            # This replacer function is called for each placeholder match found
            def replacer(match):
                ref_str = match.group(1) # The full reference, e.g., "step_0.result.files"
                
                parts = ref_str.split('.')
                step_id = parts[0]
                attribute_path = parts[1:]

                prev_output = fsm.get_step_output(step_id)
                if prev_output is None:
                    raise ValueError(f"Could not find output for referenced step: {step_id}")

                # Navigate through the output object to find the final value
                resolved_value = prev_output
                try:
                    for attr in attribute_path:
                        if isinstance(resolved_value, dict):
                            resolved_value = resolved_value.get(attr)
                        else:
                            resolved_value = getattr(resolved_value, attr)
                        if resolved_value is None:
                            raise ValueError(f"Path part '{attr}' in '{ref_str}' resolved to None.")
                except (AttributeError, KeyError, TypeError) as e:
                    raise ValueError(f"Failed to resolve path '{'.'.join(attribute_path)}' in output of {step_id}. Error: {e}")
                
                # If the entire original string was a placeholder, return the raw object (like a list).
                # Otherwise, it's embedded in a larger string, so convert it to a string.
                if match.group(0) == params:
                    return resolved_value
                else:
                    return str(resolved_value)

            # Use a temporary variable to hold the result of re.sub
            # re.sub can fail if the input `params` is not a string, which can happen
            # if a previous replacement in a dictionary comprehension returns a non-string.
            # This logic is now safer.
            if re.search(placeholder_pattern, params):
                # The crucial check: Is the entire string just one placeholder?
                match = re.fullmatch(placeholder_pattern, params)
                if match:
                    # If yes, resolve it directly without re.sub to preserve its type (e.g., list)
                    return replacer(match)
                else:
                    # If no, it's embedded in a string, so use re.sub for substitution
                    return re.sub(placeholder_pattern, replacer, params)
            
            return params
        else:
            return params

    def _run_tool_once(self, tool_name: str, params: dict) -> Dict[str, Any]:
        """
        Execute a single tool with the provided params, passing model configuration.
        """
        if tool_name == "llm_response_async":
            tool_fn = self.registry.get_function(
                tool_name,
                local_model_name=self.planner.local_model_name if self.planner else None,
                api_model_name=self.planner.api_model_name if self.planner else None
            )
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
