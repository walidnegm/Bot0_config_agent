"""
agent/prompt_builder.py
Builds and manages prompts for the tool-calling agent.
Prompts are populated with the registered tools at runtime (Jinja2).
"""

from typing import Any, Dict, List, Optional
from agent.llm_manager import LLMManager
from tools.tool_registry import ToolRegistry
from prompts.load_agent_prompts import (
    load_planner_prompts,
    load_evaluator_prompts,
    load_intent_classifier_prompts,
)

class PromptBuilder:
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize PromptBuilder with an LLMManager and ToolRegistry.

        Args:
            llm_manager (Optional[LLMManager]): Manager for local or API LLM calls.
            tool_registry (Optional[ToolRegistry]): Registry of available tools.
        """
        self.llm_manager = llm_manager
        self.tool_registry = tool_registry or ToolRegistry()
        self.is_local = bool(llm_manager.local_model) if llm_manager else False

    def build_planner_prompt(self, instruction: str) -> List[Dict[str, str]]:
        """
        Build the planner prompt with available tools injected.
        """
        prompt_dict = load_planner_prompts(
            user_task=instruction,
            tools=self.tool_registry.get_all(),
            local_model=self.is_local,
        )
        return [
            {"role": "system", "content": prompt_dict["system_prompt"]},
            {"role": "user", "content": prompt_dict["main_planner_prompt"]},
        ]

    def build_evaluator_prompt(self, task: str, response: Any) -> List[Dict[str, str]]:
        """
        Build the evaluator prompt for scoring a task response.
        """
        prompt_dict = load_evaluator_prompts(task=task, response=str(response))
        return [
            {"role": "system", "content": prompt_dict["system_prompt"]},
            {"role": "user", "content": prompt_dict["user_prompt_template"]},
        ]

    def build_intent_classifier_prompt(self, instruction: str) -> List[Dict[str, str]]:
        """
        Build the intent classifier prompt (task_decomposition or describe_only).
        """
        prompt_dict = load_intent_classifier_prompts(user_task=instruction)
        return [
            {"role": "system", "content": prompt_dict["system_prompt"]},
            {"role": "user", "content": prompt_dict["user_prompt_template"]},
        ]
