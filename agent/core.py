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
from tools.workbench.tool_registry import ToolRegistry
from tools.workbench.tool_models import ToolResult, ToolResults

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

    def handle_instruction(self, instruction: str) -> ToolResults:
        """
        Process a user instruction: plan tool usage, execute tools, and return results.

        Args:
            instruction (str): Natural language instruction from the user.

        Returns:
            List[Dict[str, Any]]: List of result dicts for each executed tool step.
        """
        logger.info(f"[AgentCore] 🧠 Received instruction: {instruction}")

        try:
            logger.debug("[AgentCore] 🧭 Calling planner.plan_async()…")

            # Create an event loop with asyncio.run(), runs the coroutine to completion,
            # then closes the loop and returns the result.
            plan = asyncio.run(self.planner.plan_async(instruction))
            executor = ToolChainExecutor(plan=plan)
            logger.debug("[AgentCore] ✅ Plan generated.")

            logger.debug("[AgentCore] 🚀 Executing plan…")

            tool_results = executor.run_plan_with_fsm()
            logger.debug("[AgentCore] ✅ Execution complete.")

            return tool_results

        # ... inside AgentCore.handle_instruction
        except Exception as e:
            logger.error(f"[AgentCore] Planner/Executor error: {e}", exc_info=True)
            # Return a standardized error result for consistent downstream handling
            error_result = ToolResult(
                step_id="step_0",
                tool="planner",
                params={},
                status=StepStatus.ERROR,
                message=str(e),
                result=None,
                state=StepState.FAILED,
            )
            return ToolResults(results=[error_result])
