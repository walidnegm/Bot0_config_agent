"""
agent/planner.py
Main planner module for generating tool chains from user instructions.
Uses an LLM (local or API) to produce a structured JSON plan, with robust
parsing and fallback handling.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agent_models.agent_models import ToolCall, ToolChain
from agent_models.step_status import StepStatus
from agent_models.step_state import StepState
from agent.prompt_builder import PromptBuilder
from prompts.load_agent_prompts import load_planner_prompts
from tools.tool_models import ToolResult, ToolResults
from tools.tool_registry import ToolRegistry
from agent.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class Planner:
    def __init__(
        self,
        local_model: Optional[str] = None,
        api_model: Optional[str] = None,
    ):
        """
        Initialize the Planner with an LLM (local or API).

        Args:
            local_model (Optional[str]): Name of local LLM model to use.
            api_model (Optional[str]): Name of API/cloud model to use.
                Exactly one of these should be set.
        """
        if (local_model and api_model) or (not local_model and not api_model):
            raise ValueError("Exactly one of local_model or api_model must be specified.")

        self.local_model = local_model
        self.api_model = api_model
        self.is_local = bool(local_model)
        logger.info(
            f"[Planner] ⚙️ Using {'local' if self.is_local else 'API'} LLM: {local_model or api_model}"
        )
        self.llm_manager = LLMManager(
            local_model=local_model,
            api_model=api_model,
        )
        self.registry = ToolRegistry()
        self.prompt_builder = PromptBuilder(
            llm_manager=self.llm_manager,
            tool_registry=self.registry,
        )

    async def plan_async(self, instruction: str) -> ToolChain:
        """
        Generate a tool chain plan for the given instruction.

        Args:
            instruction (str): User instruction to process.

        Returns:
            ToolChain: The validated tool chain plan.

        Raises:
            ValueError: If the plan cannot be generated or validated.
        """
        try:
            # Load the planner prompt with available tools
            prompt_dict = load_planner_prompts(
                user_task=instruction,
                tools=self.registry.get_all(),
                local_model=self.is_local,
            )
            system_prompt = prompt_dict["system_prompt"]
            main_prompt = prompt_dict["main_planner_prompt"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": main_prompt},
            ]

            # Call the LLM
            resp = await self.dispatch_llm_async(messages=messages)
            if not isinstance(resp, dict) or "text" not in resp:
                logger.error(f"[Planner] Unexpected LLM response format: {type(resp)} {resp}")
                raise ValueError("Failed to generate plan: Invalid LLM response format")
            raw_plan = str(resp.get("text", ""))
            logger.debug(f"[Planner] Raw LLM response: {raw_plan}")
            if not raw_plan:
                logger.error("[Planner] Empty LLM response")
                raise ValueError("Failed to generate plan: Empty LLM response")

            # Extract and validate the tool plan
            from prompts.prompt_utils import extract_and_validate_plan

            plan = extract_and_validate_plan(
                raw_text=raw_plan,
                allowed_tools=[tool["name"] for tool in self.registry.get_all()],
                use_sentinel=True,
            )

            # Validate subdirectory-specific tasks
            if "list_project_files" in [item.get("tool") for item in plan] and re.search(r"\./\w+|agent\b|tools\b|folder", instruction, re.IGNORECASE):
                for item in plan:
                    if item.get("tool") == "list_project_files" and "root" not in item.get("params", {}):
                        logger.warning("[Planner] Subdirectory mentioned in instruction but 'root' not set in list_project_files params")
                        match = re.search(r"\./(\w+)|(\w+)\s+folder", instruction, re.IGNORECASE)
                        directory = match.group(1) or match.group(2) or "agent"
                        item["params"]["root"] = directory

            if not plan:
                logger.error("[Planner] No valid tool calls extracted from LLM response")
                raise ValueError("Failed to generate plan: No valid tool calls extracted")

            # Convert to Pydantic model
            try:
                return ToolChain(steps=[ToolCall(**item) for item in plan])
            except Exception as e:
                logger.error(f"[Planner] Failed to parse plan as ToolChain: {e}")
                raise ValueError(f"Failed to generate plan: Invalid tool chain format: {e}")

        except Exception as e:
            logger.error(f"[Planner] Failed to generate plan: {e}", exc_info=True)
            raise ValueError(f"Failed to generate plan: {e}")

    async def dispatch_llm_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch the LLM call to the appropriate backend (local or API).

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature for generation.
            generation_kwargs: Optional additional generation parameters.

        Returns:
            Dict with "text" key containing the LLM response.

        Raises:
            ValueError: If the LLM client is not initialized or the call fails.
        """
        try:
            resp = await self.llm_manager.generate_async(
                messages=messages,
                temperature=temperature,
                **(generation_kwargs or {}),
            )
            return {"text": resp or ""}
        except Exception as e:
            logger.error(f"[Planner] LLM dispatch failed: {e}", exc_info=True)
            raise ValueError(f"LLM dispatch failed: {e}")

    async def evaluate_async(self, task: str, response: Any) -> Dict[str, Any]:
        """
        Evaluate a task response using the LLM.

        Args:
            task: The original user task/instruction.
            response: The response to evaluate (e.g., tool output).

        Returns:
            Dict with evaluation results (e.g., {"score": float, "reasoning": str}).

        Raises:
            ValueError: If the evaluation fails.
        """
        try:
            from prompts.load_agent_prompts import load_evaluator_prompts

            prompt_dict = load_evaluator_prompts(task=task, response=str(response))
            system_prompt = prompt_dict["system_prompt"]
            user_prompt = prompt_dict["user_prompt_template"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            resp = await self.dispatch_llm_async(messages=messages)
            raw_eval = (resp or {}).get("text", "")
            if not raw_eval:
                logger.error("[Planner] Empty evaluation response")
                raise ValueError("Evaluation failed: Empty response")

            from agent_models.llm_response_validators import clean_and_extract_json

            try:
                eval_dict = clean_and_extract_json(raw_eval)
                if not isinstance(eval_dict, dict):
                    logger.error("[Planner] Evaluation response is not a dict: %s", eval_dict)
                    raise ValueError("Evaluation failed: Invalid response format")
                return eval_dict
            except Exception as e:
                logger.error(f"[Planner] Failed to parse evaluation: {e}")
                raise ValueError(f"Evaluation failed: Invalid response format: {e}")

        except Exception as e:
            logger.error(f"[Planner] Evaluation failed: {e}", exc_info=True)
            raise ValueError(f"Evaluation failed: {e}")
