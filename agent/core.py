"""
agent/core.py
Core agent logic: initializes tool registry, planner, and executor, and processes instructions.
Supports both local and cloud (API) LLM model selection via explicit arguments.
"""

import logging
from typing import Optional, List, Dict, Any
import asyncio
from agent.executor import ToolExecutor
from agent.planner import Planner
from tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentCore:
    """
    Main agent class that wires together tool registry, planner, and executor.
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
        self.executor = ToolExecutor()
        logger.info("[AgentCore] ✅ Initialization complete.")

    def handle_instruction(
        self, instruction: str
    ) -> List[Dict[str, Any]]:  # ☑️ updated this to allow async function calling
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
            logger.debug("[AgentCore] ✅ Plan generated.")

            logger.debug("[AgentCore] 🚀 Executing plan…")
            results = self.executor.execute_plan(plan=plan)
            logger.debug("[AgentCore] ✅ Execution complete.")

            return results

        except Exception as e:
            logger.error(f"[AgentCore] Planner error: {e}", exc_info=True)
            return [{"tool": "planner", "status": "error", "message": str(e)}]
