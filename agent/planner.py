"""agent/planner.py
Tool planning: Converts user instructions into a validated ToolChain (Pydantic) plan
using a local or cloud LLM, driven by a Jinja2-rendered YAML prompt file.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional, Type, Union
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
import yaml
# Registry and agent models
from tools.tool_registry import ToolRegistry
from agent_models.agent_models import (
    CodeResponse,
    JSONResponse,
    TextResponse,
    ToolCall,
    ToolChain,
)
# Local/Cloud LLM bridges
from agent.llm_manager import get_llm_manager
from utils.llm_api_async import (
    call_anthropic_api_async,
    call_gemini_api_async,
    call_openai_api_async,
    call_llm_api_async,
)
# Prompt loading helpers and paths
from configs.paths import AGENT_PROMPTS
from prompts.prompt_utils import extract_and_validate_plan  # Updated import
from configs.api_models import validate_api_model_name, get_llm_provider

logger = logging.getLogger(__name__)

class Planner:
    """
    Planner: Generates a validated ToolChain from user instructions.
    Chooses between local and API model based on initialization.
    """
    def __init__(
        self,
        local_model_name: Optional[str] = None,
        api_model_name: Optional[str] = None,
    ):
        if not (local_model_name or api_model_name):
            raise ValueError(
                "You must specify at least one of local_model_name or api_model_name."
            )
        self.tool_registry = ToolRegistry()
        self.local_model_name = local_model_name
        self.api_model_name = api_model_name
        if self.api_model_name:
            logger.info(f"[Planner] ⚙️ Using API/cloud LLM: {self.api_model_name}")
        else:
            logger.info(f"[Planner] ⚙️ Using local LLM: {self.local_model_name}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def plan_async(self, instruction: str) -> ToolChain:
        """
        Asynchronously generates a plan (list of tool calls) from the user instruction.
        Renders the Jinja template -> YAML -> builds prompts -> calls LLM -> extracts plan.
        """
        instruction = instruction.strip()
        # 1) Load & render the planner YAML template with available tools
        try:
            env = Environment(loader=FileSystemLoader(AGENT_PROMPTS.parent))
            template = env.get_template(AGENT_PROMPTS.name)
            rendered_yaml = template.render(
                user_task=instruction,
                tools=self.tool_registry.get_all(),
                local_model=bool(self.local_model_name)  # Pass for prompt customization
            )
            cfg = yaml.safe_load(rendered_yaml)
            if "planner" not in cfg:
                raise KeyError("Planner section not found in rendered YAML template.")
            logger.info("[Planner] Loaded and rendered main planner prompt.")
        except Exception as e:
            logger.error(f"[Planner] Failed to load or render template: {e}", exc_info=True)
            return self._fallback_llm_response(instruction)

        # 2) Build the full prompt
        system_prompt = cfg["planner"].get("system_prompt", "").strip()
        main_prompt = cfg["planner"].get("main_planner_prompt", "").strip()
        if not system_prompt or not main_prompt:
            logger.warning("[Planner] Missing system_prompt or main_planner_prompt in template.")
            return self._fallback_llm_response(instruction)
        full_prompt = f"{system_prompt}\n{main_prompt}"
        logger.debug(f"[Planner] Full rendered prompt:\n{full_prompt}")

        # 3) Call the LLM
        try:
            raw_plan = await self._plan_with_llm_async(full_prompt)
            logger.debug(f"[Planner] Raw LLM output:\n{raw_plan}")
        except Exception as e:
            logger.error(f"[Planner] LLM call failed: {e}", exc_info=True)
            return self._fallback_llm_response(instruction)

        # 4) Extract and validate the plan
        try:
            plan = self._extract_plan(raw_plan, instruction)
            return self._parse_and_validate_tool_calls(plan, instruction)
        except Exception as e:
            logger.error(f"[Planner] Plan extraction/validation failed: {e}", exc_info=True)
            return self._fallback_llm_response(instruction)

    async def dispatch_llm_async(
        self,
        user_prompt: str,
        system_prompt: str = "",
        response_type: Literal["json", "text", "code"] = "json",
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Union[JSONResponse, TextResponse, CodeResponse, str]:
        """
        Universal dispatcher for local or API LLM calls.
        """
        if self.api_model_name:
            provider = get_llm_provider(self.api_model_name)
            return await call_llm_api_async(
                provider=provider,
                model_id=self.api_model_name,
                prompt=user_prompt,
                system_prompt=system_prompt,
                response_format=response_type,
                **kwargs,
            )
        else:
            llm_manager = get_llm_manager(self.local_model_name)
            return await llm_manager.generate_async(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                expected_res_type=response_type,
                response_model=response_model,
                **kwargs,
            )

    # ---------------------------------------------------------------------
    # Internal: LLM Planning
    # ---------------------------------------------------------------------
    async def _plan_with_llm_async(self, full_prompt: str) -> str:
        """
        Call the LLM to generate a JSON plan (array of tool calls).
        Returns the raw text response (JSON array expected).
        """
        logger.debug(f"[Planner] Sending prompt to LLM:\n{full_prompt}")
        response = await self.dispatch_llm_async(
            user_prompt=full_prompt,
            system_prompt="",
            response_type="json",
            response_model=JSONResponse,
            max_tokens=1024,
            temperature=0.1 if self.local_model_name else 0.2,  # Lower for local to reduce hallucination
            do_sample=False if self.local_model_name else True  # No sampling for local
        )
        # Handle both direct string and wrapped JSONResponse
        raw_text = (
            response.text
            if isinstance(response, JSONResponse)
            else str(response)
        )
        logger.debug(f"[Planner] Raw LLM response:\n{raw_text}")
        return raw_text

    def _extract_plan(self, raw_text: str, fallback_prompt: str) -> List[Dict[str, Any]]:
        """
        Extract a list of tool calls from raw LLM output.
        Steps:
          1) Use extract_and_validate_plan to get validated tool calls.
          2) If nothing valid, fall back to the LLM response tool with the user instruction.
        """
        plan = extract_and_validate_plan(
            raw_text=raw_text,
            allowed_tools=[tool["name"] for tool in self.tool_registry.get_all()],
            use_sentinel=True,
            sentinel="FINAL_JSON"
        )
        if plan:
            logger.info("[Planner] ✅ Extracted valid JSON tool plan.")
            return plan
        logger.warning(
            "[Planner] No valid JSON plan found. Falling back to LLM response."
        )
        return [{"tool": "llm_response_async", "params": {"prompt": fallback_prompt}}]

    def _parse_and_validate_tool_calls(
        self, tool_calls: List[Dict[str, Any]], instruction: str
    ) -> ToolChain:
        """
        Parses and validates a list of tool call dictionaries against the registry.
        Unknown tools are skipped; if nothing valid, fall back to LLM response.
        """
        validated_steps: List[ToolCall] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = call.get("tool")
            params = call.get("params", {})
            if not tool_name or not isinstance(params, dict):
                continue
            if not self.tool_registry.has_tool(tool_name):
                logger.warning(
                    "[Planner] Plan contained an unknown tool: '%s'. Skipping.", tool_name
                )
                continue
            validated_steps.append(ToolCall(tool=tool_name, params=params))
        if not validated_steps:
            logger.warning(
                "[Planner] No valid tool calls found in the plan. Falling back."
            )
            return self._fallback_llm_response(instruction)
        logger.info(
            "[Planner] Returning ToolChain with %d validated tool call(s).",
            len(validated_steps),
        )
        return ToolChain(steps=validated_steps)

    # ---------------------------------------------------------------------
    # Internal: Fallback
    # ---------------------------------------------------------------------
    def _fallback_llm_response(self, instruction: str) -> ToolChain:
        """A generic catch-all fallback tool call."""
        fallback_call = ToolCall(
            tool="llm_response_async", params={"prompt": instruction}
        )
        return ToolChain(steps=[fallback_call])
