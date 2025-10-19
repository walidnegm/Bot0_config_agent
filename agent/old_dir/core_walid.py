"""
agent/core.py
Core agent logic: initializes tool registry, planner, and executor, and processes instructions.
Supports both local and cloud (API) LLM model selection via explicit arguments.
"""
import logging
from typing import Optional, Dict, Union
import asyncio
from agent.planner import Planner
from agent.tool_chain_executor import ToolChainExecutor
from agent_models.step_state import StepState
from agent_models.step_status import StepStatus
from tools.tool_registry import ToolRegistry
from tools.tool_models import ToolResult, ToolResults
logger = logging.getLogger(__name__)

class AgentCore:
    """
    Main agent class that wires together tool_registry, planner, and tool_chain_executor.
    """
    def __init__(
        self,
        local_model_name: Optional[str] = None,
        api_model_name: Optional[str] = None,
    ):
        """
        Initialize AgentCore with local or API model.
        Args:
            local_model_name (Optional[str]): Name of local LLM model to use.
            api_model_name (Optional[str]): Name of API/cloud model to use.
                Exactly one of these should be set.
        """
        logger.info("[AgentCore] 🔧 Initializing ToolRegistry, Planner, and Executor…")
        self.registry = ToolRegistry()
        self.planner = Planner(
            local_model_name=local_model_name, api_model_name=api_model_name
        )
        logger.info("[AgentCore] ✅ Initialization complete.")


    # In agent/core.py
    # In agent/core.py

    def handle_instruction(self, instruction: str) -> ToolResults:
        """
        Process a user instruction: plan tool usage, execute tools, and return results.
        """
        logger.info(f"[AgentCore] 🧠 Received instruction: {instruction}")
        try:
            logger.debug("[AgentCore] 🧭 Calling planner.plan_async()…")
            plan = asyncio.run(self.planner.plan_async(instruction))

            # If the plan is empty or just a fallback, handle it gracefully.
            if not plan or not plan.steps:
                logger.warning("[AgentCore] Planner returned an empty plan. No tools to execute.")
                # You can decide to return an empty result or a specific message
                return ToolResults(results=[])

            executor = ToolChainExecutor(plan=plan, planner=self.planner)
            logger.debug("[AgentCore] ✅ Plan generated: %s", plan)
            logger.debug("[AgentCore] 🚀 Executing plan…")

            # ##########################################################################
            # FIXED LINE: Added the required 'plan' argument back to the call.
            # ##########################################################################
            tool_results = executor.run_plan_with_fsm(plan)

            logger.debug("[AgentCore] ✅ Execution complete.")
            return tool_results

        except Exception as e:
            logger.error(f"[AgentCore] Critical error in handle_instruction: {e}", exc_info=True)
            error_result = ToolResult(
                step_id="step_0",
                tool="planner",
                params={},
                status=StepStatus.ERROR,
                message=f"A critical error occurred: {e}",
                result=None,
                state=StepState.FAILED,
            )
            return ToolResults(results=[error_result])
