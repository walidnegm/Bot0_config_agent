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

    def _resolve_step_dependencies(self, params: Any, fsm: ToolChainFSM) -> Any:
        """
        Recursively resolve placeholders in params (str, dict, list).
        """
        placeholder_pattern = r"<step_(\d+)(?:\.(.+?))?>"

        def replacer(match):
            step_id = f"step_{match.group(1)}"
            attribute_path = match.group(2).split('.') if match.group(2) else []
            prev_output = fsm.get_step_output(step_id)
            if prev_output is None:
                logger.warning(f"No output available for referenced step: {step_id}. Using empty string as fallback.")
                return ""
            value = prev_output
            try:
                for attr in attribute_path:
                    if isinstance(value, dict):
                        value = value.get(attr)
                    else:
                        value = getattr(value, attr)
                    if value is None:
                        logger.warning(f"Attribute '{attr}' resolved to None in '{match.group(0)}'. Using empty string.")
                        return ""
                return value
            except (AttributeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to resolve path in '{match.group(0)}': {e}. Using empty string.")
                return ""

        if isinstance(params, str):
            match = re.fullmatch(placeholder_pattern, params)
            if match:
                return replacer(match)
            return re.sub(placeholder_pattern, replacer, params)
        elif isinstance(params, dict):
            return {k: self._resolve_step_dependencies(v, fsm) for k, v in params.items()}
        elif isinstance(params, list):
            return [self._resolve_step_dependencies(v, fsm) for v in params]
        else:
            return params

    def _run_tool_once(self, tool_name: str, params: dict) -> Dict[str, Any]:
        """
        Execute a single tool with the provided params, passing model configuration.
        """
        if tool_name == "llm_response_async":
            tool_fn = self.registry.get_function(
                tool_name,
                planner=self.planner
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
