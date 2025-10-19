"""
agent/core.py
Core agent logic (pure MCP mode).
Initializes planner and executor using dynamically injected prompts and MCP-discovered tools.
"""

import logging
import asyncio
from typing import Optional, Dict, Any

from agent.planner import Planner
from agent.tool_chain_executor import ToolChainExecutor
from agent_models.step_state import StepState
from agent_models.step_status import StepStatus
from tools.tool_models import ToolResult, ToolResults

logger = logging.getLogger(__name__)


class AgentCore:
    """
    Main agent class that wires together planner and executor (MCP-based).
    """

    def __init__(
        self,
        local_model_name: Optional[str] = None,
        api_model_name: Optional[str] = None,
        planner_prompts: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AgentCore with local or API model and injected planner prompts.
        Args:
            local_model_name: Optional name of local LLM model.
            api_model_name: Optional name of API/cloud model.
            planner_prompts: Pre-rendered planner section (injected via MCP tool discovery).
        """
        logger.info("[AgentCore] 🔧 Initializing Planner and Executor (pure MCP mode)…")

        # No ToolRegistry — MCP is authoritative
        self.planner = Planner(
            local_model=local_model_name,
            api_model=api_model_name,
            prompts=planner_prompts,
        )

        logger.info("[AgentCore] ✅ Initialization complete.")

    def handle_instruction(self, instruction: str) -> ToolResults:
        """
        Process a user instruction: plan tool usage, execute tools, and return results.
        """
        logger.info(f"[AgentCore] 🧠 Received instruction: {instruction}")
        try:
            logger.debug("[AgentCore] 🧭 Calling planner.plan_async()…")
            plan = asyncio.run(self.planner.plan_async(instruction))

            if not plan or not plan.steps:
                logger.warning("[AgentCore] Planner returned an empty plan. No tools to execute.")
                return ToolResults(results=[])

            executor = ToolChainExecutor(plan=plan, planner=self.planner)
            logger.debug("[AgentCore] ✅ Plan generated: %s", plan)
            logger.debug("[AgentCore] 🚀 Executing plan…")

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

