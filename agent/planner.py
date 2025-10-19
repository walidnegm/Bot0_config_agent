"""
agent/planner.py
Planner that generates tool chains dynamically using MCP-discovered tools.
No hardcoded ToolRegistry references — MCP tools are injected through planner_prompts.
"""

import logging
import re
import json
from typing import Any, Dict, List, Optional

from agent_models.agent_models import ToolCall, ToolChain
from agent.llm_manager import LLMManager
from agent.prompt_builder import PromptBuilder
from agent_models.step_status import StepStatus
from agent_models.step_state import StepState
from prompts.load_agent_prompts import load_planner_prompts

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"^<step_(\d+)(?:\.[^>]*)?>$")


def _step_index_from_placeholder(s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    m = _PLACEHOLDER_RE.match(s.strip())
    return int(m.group(1)) if m else None


class Planner:
    def __init__(
        self,
        local_model: Optional[str] = None,
        api_model: Optional[str] = None,
        prompts: Optional[Dict[str, Any]] = None,
    ):
        if (local_model and api_model) or (not local_model and not api_model):
            raise ValueError("Exactly one of local_model or api_model must be specified.")

        self.local_model = local_model
        self.api_model = api_model
        self.is_local = bool(local_model)
        self.prompts = prompts  # dynamically injected planner section

        logger.info(f"[Planner] ⚙️ Using {'local' if self.is_local else 'API'} LLM: {local_model or api_model}")

        self.llm_manager = LLMManager(local_model=local_model, api_model=api_model)
        self.prompt_builder = PromptBuilder(llm_manager=self.llm_manager)

    async def dispatch_llm_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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

    async def plan_async(self, instruction: str) -> ToolChain:
        try:
            # ✅ Use pre-rendered prompts if available
            if self.prompts:
                prompt_dict = self.prompts
            else:
                logger.warning("[Planner] No prompts provided — falling back to static template.")
                prompt_dict = load_planner_prompts(user_task=instruction, tools=[], local_model=self.is_local)

            system_prompt = prompt_dict["system_prompt"]
            main_prompt = prompt_dict["main_planner_prompt"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": main_prompt},
            ]

            resp = await self.dispatch_llm_async(messages=messages)
            raw_plan = str(resp.get("text", "")).strip()

            if not raw_plan:
                raise ValueError("Planner LLM returned empty response")

            from prompts.prompt_utils import extract_and_validate_plan
            plan = extract_and_validate_plan(raw_text=raw_plan, use_sentinel=True)

            logger.info(f"[Planner] Generated plan: {json.dumps(plan, indent=2)}")

            return ToolChain(steps=[ToolCall(**item) for item in plan])

        except Exception as e:
            logger.error(f"[Planner] Failed to generate plan: {e}", exc_info=True)
            raise ValueError(f"Failed to generate plan: {e}")

