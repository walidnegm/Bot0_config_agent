"""
agent/planner.py
Main planner module for generating tool chains from user instructions.
Uses an LLM (local or API) to produce a structured JSON plan, with robust
parsing and fallback handling.
"""
import logging
import re
import json
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

# Accepts "<step_0>" and "<step_0.result>" (and optional dotted fields after the index)
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
    ):
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
            resp = await self.dispatch_llm_async(messages=messages)
            if not isinstance(resp, dict) or "text" not in resp:
                logger.error(f"[Planner] Unexpected LLM response format: {type(resp)} {resp}")
                raise ValueError("Failed to generate plan: Invalid LLM response format")
            raw_plan = str(resp.get("text", ""))
            logger.debug(f"[Planner] Raw LLM response: {raw_plan}")
            if not raw_plan:
                logger.error("[Planner] Empty LLM response")
                raise ValueError("Failed to generate plan: Empty LLM response")

            from prompts.prompt_utils import extract_and_validate_plan
            plan = extract_and_validate_plan(
                raw_text=raw_plan,
                allowed_tools=[tool["name"] for tool in self.registry.get_all()],
                use_sentinel=True,
            )

            # ---- Safety: coerce known param shapes right after parse (helps if LLM used JSON strings) ----
            def _coerce_param_shapes(item: Dict[str, Any]) -> None:
                tool = item.get("tool")
                params = item.get("params", {})
                # read_files.path should be a list (or a single string path)
                if tool == "read_files" and "path" in params:
                    p = params["path"]
                    if isinstance(p, str):
                        s = p.strip()
                        # If it looks like a JSON array, parse to list
                        if s.startswith("[") and s.endswith("]"):
                            try:
                                parsed = json.loads(s)
                                if isinstance(parsed, list):
                                    params["path"] = parsed
                            except Exception:
                                pass
                    # If it's not list, make it a singleton list (tools can still handle it)
                    if not isinstance(params["path"], list):
                        params["path"] = [params["path"]]

                # aggregate_file_content.steps and summarize_* .files should be lists
                if tool in {"aggregate_file_content", "summarize_config", "summarize_files"}:
                    key = "steps" if tool == "aggregate_file_content" else "files"
                    if key in params and not isinstance(params[key], list):
                        params[key] = [params[key]]

            for item in plan:
                _coerce_param_shapes(item)

            # ---- Validate structure and placeholders (no forward references) ----
            required_params = {
                "list_project_files": ["root"],
                "read_files": ["path"],
                "aggregate_file_content": ["steps"],
                "llm_response_async": ["prompt"],
                "find_dir_size": ["root"],
                "find_dir_structure": ["root"],
                "find_file_by_keyword": ["root", "keyword"],
                "locate_file": ["filename"],
                "make_virtualenv": ["path"],
                "set_scope": ["scope"],
                "echo_message": ["message"],
                "retrieval_tool": ["query"],
                "seed_parser": ["data"],
                "summarize_config": ["files"],
                "summarize_files": ["files"],
            }

            def _is_forward_ref(idx: int, current_step: int) -> bool:
                return idx >= current_step

            def _check_params(p: Any, current_step: int):
                if isinstance(p, dict):
                    for v in p.values():
                        _check_params(v, current_step)
                elif isinstance(p, list):
                    for v in p:
                        _check_params(v, current_step)
                elif isinstance(p, str):
                    idx = _step_index_from_placeholder(p)
                    if idx is not None and _is_forward_ref(idx, current_step):
                        logger.warning(f"[Planner] Invalid or forward-referencing placeholder {p} in step_{current_step}")

            for i, item in enumerate(plan):
                tool = item.get("tool")
                params = item.get("params", {})
                if tool not in required_params:
                    logger.warning(f"[Planner] Unknown tool {tool} in plan")
                    continue
                for param in required_params.get(tool, []):
                    if param not in params:
                        logger.warning(f"[Planner] Missing required parameter '{param}' for tool {tool} in step_{i}")
                        if tool == "list_project_files" and param == "root":
                            params["root"] = "."
                _check_params(params, i)

            logger.info(f"[Planner] Generated plan: {json.dumps(plan, indent=2)}")
            try:
                return ToolChain(steps=[ToolCall(**item) for item in plan])
            except Exception as e:
                logger.error(f"[Planner] Failed to parse plan as ToolChain: {e}")
                raise ValueError(f"Failed to generate plan: Invalid tool chain format: {e}")
        except Exception as e:
            logger.error(f"[Planner] Failed to generate plan: {e}", exc_info=True)
            raise ValueError(f"Failed to generate plan: {e}")

    async def evaluate_async(self, task: str, response: Any) -> Dict[str, Any]:
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

