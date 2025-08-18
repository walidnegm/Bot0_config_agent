"""
agent/planner.py

Tool planning: Converts user instructions into a validated ToolChain (Pydantic) plan
using either a local or cloud LLM. Handles prompt construction, intent detection,
single vs multi-step planning, and plan validation.

- All public APIs always return ToolChain (never dict, list, or ToolCall).
"""
from configs.paths import AGENT_PROMPTS
import re
import logging
from typing import Optional, Union, Type, Literal, List, Dict, Any
import asyncio
from pydantic import BaseModel
from tools.tool_registry import ToolRegistry

# Intent classification
from agent.intent_classifiers import (
    classify_describe_only_async,
    classify_task_decomposition_async,
)

# Local model manager + API bridges
from agent.llm_manager import get_llm_manager
from utils.llm_api_async import (
    call_openai_api_async,
    call_anthropic_api_async,
)

# Core agent models
from agent_models.agent_models import (
    JSONResponse,
    TextResponse,
    CodeResponse,
    ToolCall,
    ToolChain,
)

from prompts.load_agent_prompts import load_planner_prompts, load_summarizer_prompt

from configs.api_models import (
    GPT_4_1,
    GPT_4_1_MINI,
    GPT_4_1_NANO,
    CLAUDE_HAIKU,
    CLAUDE_SONNET_4,
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    GEMINI_MODELS,
)
from configs.api_models import get_llm_provider

# Robust JSON plan extraction helpers / constants
from prompts.prompt_utils import (
    extract_all_json_arrays,
    safe_parse_json_array,
    FINAL_SENTINEL,  # usually "FINAL_JSON"
)

logger = logging.getLogger(__name__)

OPENAI = "openai"
ANTHROPIC = "anthropic"
RESPONSE_TYPE = ["code", "json", "text"]
API_LLM_MAX_TOKEN_DEFAULT: int = 1024
API_LLM_TEMPERATURE_DEFAULT: float = 0.2


def validate_api_model_name(model_name: str) -> None:
    """Raises ValueError if model_name is not a known cloud API model."""
    if (
        model_name not in OPENAI_MODELS
        and model_name not in ANTHROPIC_MODELS
        and model_name not in GEMINI_MODELS
    ):
        raise ValueError(f"Unknown or unsupported API model: {model_name}")


class Planner:
    """
    Planner: Generates a validated ToolChain from user instructions.
    Always returns a Pydantic ToolChain instance (never dict/list/ToolCall).
    Supports both local and cloud LLMs (async-safe).
    """

    def __init__(
        self,
        local_model_name: Optional[str] = None,
        api_model_name: Optional[str] = None,
    ):
        # Safeguard: both cannot be None
        if not (local_model_name or api_model_name):
            raise ValueError(
                "You must specify at least one of local_model_name or api_model_name."
            )

        self.tool_registry = ToolRegistry()
        self.local_model_name = local_model_name
        self.api_model_name = api_model_name

        if self.api_model_name:
            logger.info(f"[Planner] ⚙️ Using API/cloud LLM: {self.api_model_name}")
        elif self.local_model_name:
            logger.info(f"[Planner] ⚙️ Using local LLM: {self.local_model_name}")
        else:
            logger.error("[Planner] No LLM model specified!")
            raise ValueError("Must specify either local_model_name or api_model_name.")

        # Param alias normalization (applied before validation)
        self.param_aliases = {
            "file_path": "path",
            "filename": "path",
            "filepath": "path",
            "root": "path",          # IMPORTANT: fix for tools expecting `path`
            "dir": "path",
        }

    async def dispatch_llm_async(
        self,
        user_prompt: str,
        system_prompt: str,
        response_type: Literal["json", "text", "code"] = "text",
        response_model: Type[BaseModel] = TextResponse,
    ) -> Union[
        JSONResponse,
        TextResponse,
        CodeResponse,
        ToolCall,
        ToolChain,
    ]:
        """
        Asynchronously routes a language model call to either a remote API LLM (e.g., OpenAI,
        Anthropic) or a local model (via LLMManager) using an executor thread.
        """
        # API LLMs: use async (faster)
        if self.api_model_name:
            full_prompt = "\n".join([system_prompt or "", user_prompt or ""])
            result = await self._call_api_llm_async(
                prompt=full_prompt,
                expected_res_type=response_type,
                response_model=response_model,
            )
            return result  # return pydantic model

        # Local LLMs (sync -> executor)
        elif self.local_model_name:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._call_local_llm(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    expected_res_type=response_type,
                    response_model=response_model,
                ),
            )
            return result
        else:
            raise ValueError(
                f"Unknown model name: {self.api_model_name or self.local_model_name}"
            )

    async def plan_async(self, instruction: str) -> ToolChain:
        """
        Asynchronously generate a plan (list of tool calls) from the user instruction.
        Uses the configured local or API LLM.

        Returns:
            ToolChain: Validated, normalized sequence of tool calls.
        """
        instruction = instruction.strip()

        # 1) Task decomposition: single or multi-step
        task_decomp = await classify_task_decomposition_async(instruction, self)
        logger.info(f"[Planner] Task decomposition result: {task_decomp}")
        is_single = bool(re.search(r"\bsingle[- ]?step\b", task_decomp, re.IGNORECASE))

        # 2) Load planner prompt templates (Jinja2 YAML-based)
        prompts_dic = load_planner_prompts(
            user_task=instruction,
            template_path=AGENT_PROMPTS,   # force Jinja template
        )

        # sanity: fail fast with a clear message if the wrong template gets loaded
        required_keys = [
            "system_prompt",
            "select_single_tool_prompt",
            "select_multi_tool_prompt",
            "return_json_only_prompt",
        ]
        missing = [k for k in required_keys if k not in prompts_dic]
        if missing:
            raise KeyError(
                f"Planner prompt missing keys {missing}. "
                f"Template in use: {AGENT_PROMPTS}"
            )

        logger.info(f"[Planner] Loaded planner keys: {sorted(prompts_dic.keys())}")

        # 3) Combine prompt templates into user prompt (with "return JSON only" section)
        def combine_prompts(select_tool_prompt: str) -> str:
            user_task_line = (
                prompts_dic.get("user_task_prompt") or f"User task: {instruction}"
            )
            user_prompt = (
                select_tool_prompt.strip()
                + "\n"
                + prompts_dic["return_json_only_prompt"].strip()
                + "\n"
                + user_task_line.strip()
            )
            return user_prompt

        system_prompt = prompts_dic.get("system_prompt", "")
        if is_single:
            user_prompt = combine_prompts(prompts_dic["select_single_tool_prompt"])
        else:
            user_prompt = combine_prompts(prompts_dic["select_multi_tool_prompt"])

        # 4) Always request TEXT and parse JSON arrays ourselves.
        text_result = await self.dispatch_llm_async(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_type="text",
            response_model=TextResponse,
        )

        # 5) Extract a single JSON array plan from the raw text
        if isinstance(text_result, TextResponse):
            raw_text = (
                getattr(text_result, "text", "")
                or getattr(text_result, "content", "")
                or ""
            )
        else:
            # ultra-defensive: if a provider wrapped differently
            raw_text = str(text_result)

        plan_list = self._extract_single_plan_from_text(
            raw_text=raw_text,
            prefer_single=is_single,
            fallback_prompt=instruction,  # use the user's ask, not the planner prompt
        )

        # 6) Validate + normalize via tool registry into a ToolChain
        tool_chain = self._parse_and_validate_tool_calls(plan_list, instruction)

        return tool_chain

    # todo: deprecate later; to_describe intent should be handled by standard tool chain
    async def _classify_intent_async(self, instruction: str) -> ToolChain | str:
        try:
            return await classify_describe_only_async(instruction, self)
        except Exception as e:
            logger.warning(f"Intent classifier failed, defaulting to LLM. Error: {e}")
            return "unknown"

    async def _call_api_llm_async(
        self,
        prompt: str,
        expected_res_type: Literal["json", "text", "code"] = "text",
        response_model: Optional[Type[BaseModel]] = TextResponse,
    ) -> Union[
        JSONResponse,
        TextResponse,
        CodeResponse,
        ToolCall,
        ToolChain,
    ]:
        """
        * Async version
        Call the specified LLM API and return the response.
        """
        assert self.api_model_name, "No API model specified. Please select a model."
        validate_api_model_name(self.api_model_name)

        llm_provider = get_llm_provider(self.api_model_name)
        model_id = self.api_model_name
        max_tokens = API_LLM_MAX_TOKEN_DEFAULT
        temperature = API_LLM_TEMPERATURE_DEFAULT

        if llm_provider.lower() == "openai":
            validated_result = await call_openai_api_async(
                prompt=prompt,
                model_id=model_id,
                expected_res_type=expected_res_type,
                temperature=temperature,
                max_tokens=max_tokens,
                response_model=response_model,
            )
        elif llm_provider.lower() == "anthropic":
            validated_result = await call_anthropic_api_async(
                prompt=prompt,
                model_id=model_id,
                expected_res_type=expected_res_type,
                temperature=temperature,
                max_tokens=max_tokens,
                response_model=response_model,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        return validated_result

    def _call_local_llm(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        expected_res_type: Literal["json", "text", "code"] = "text",
        response_model: Type[BaseModel] = TextResponse,
    ) -> TextResponse | CodeResponse | JSONResponse | ToolCall | ToolChain:
        """
        Calls the local LLM with the provided prompt.
        """
        assert self.local_model_name is not None
        llm = get_llm_manager(self.local_model_name)

        validated_result = llm.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            expected_res_type=expected_res_type,
            response_model=response_model,
        )

        return validated_result  # Return pydantic model

    def _extract_single_plan_from_text(
        self,
        raw_text: str,
        prefer_single: bool,
        *,
        fallback_prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Robustly extract a JSON array plan from model text.

        Strategy:
        1) If FINAL_JSON appears: parse JSON arrays that follow it.
           - Prefer the last valid NON-EMPTY array after the LAST FINAL_JSON.
           - If only EMPTY arrays after FINAL_JSON, keep looking (Step 2).
        2) Otherwise (or if Step 1 yields nothing useful):
           - Parse the tail of the WHOLE output (e.g., last 1500 chars),
             take the LAST valid NON-EMPTY array (e.g., 'Assistant: [{"tool":...}]').
        3) If still nothing, fall back to llm_response_async with the original instruction.
        4) If prefer_single and the plan has >1 steps, keep only the first.

        Notes:
        - Valid array == list of dicts each with "tool" (str) and "params" (dict).
        - NON-EMPTY means len(array) > 0.
        """

        def _to_arrays(txt: str) -> List[List[Dict[str, Any]]]:
            arrays_raw = extract_all_json_arrays(txt) or []
            arrays: List[List[Dict[str, Any]]] = []
            for js in arrays_raw:
                arr = safe_parse_json_array(js)
                if (
                    isinstance(arr, list)
                    and all(isinstance(x, dict) and "tool" in x and "params" in x for x in arr)
                    and all(isinstance(x.get("tool"), str) and isinstance(x.get("params"), dict) for x in arr)
                ):
                    arrays.append(arr)
            return arrays

        def _pick_last_non_empty(arrays: List[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
            non_empty = [a for a in arrays if len(a) > 0]
            if non_empty:
                return non_empty[-1]
            return None

        # --- PASS 1: prefer arrays AFTER the LAST FINAL_JSON sentinel ---
        sentinel = FINAL_SENTINEL.strip()  # usually "FINAL_JSON"
        matches = list(re.finditer(rf"(?mi)^[ \t]*{re.escape(sentinel)}[ \t]*\r?$", raw_text))
        if matches:
            last = matches[-1]
            post = raw_text[last.end():]  # text after the last sentinel line
            post_arrays = _to_arrays(post)
            plan_after_final = _pick_last_non_empty(post_arrays)
            if plan_after_final is not None:
                plan = plan_after_final
                if prefer_single and len(plan) > 1:
                    logger.info("[Planner] Reducing multi-step plan to single step based on classifier.")
                    plan = [plan[0]]
                return plan
            else:
                logger.warning("[Planner] Only empty or invalid arrays found AFTER FINAL_JSON; trying whole-output tail.")

        # --- PASS 2: no good array after FINAL_JSON; try the tail of the WHOLE output ---
        tail = raw_text[-1500:] if len(raw_text) > 1500 else raw_text
        tail_arrays = _to_arrays(tail)
        plan_from_tail = _pick_last_non_empty(tail_arrays)
        if plan_from_tail is not None:
            plan = plan_from_tail
            if prefer_single and len(plan) > 1:
                logger.info("[Planner] Reducing multi-step plan to single step based on classifier (tail).")
                plan = [plan[0]]
            return plan

        # --- PASS 3: fallback ---
        logger.warning("[Planner] No valid JSON plan found; falling back to LLM response.")
        return [{"tool": "llm_response_async", "params": {"prompt": fallback_prompt}}]

    def _normalize_param_aliases(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map known alias keys to canonical keys (e.g., root -> path)."""
        if not params:
            return {}
        normalized = dict(params)
        for k, v in list(params.items()):
            if k in self.param_aliases:
                target = self.param_aliases[k]
                # Don't overwrite if canonical already present
                if target not in normalized:
                    normalized[target] = v
                    logger.debug(f"[Planner] Alias normalized: '{k}' -> '{target}' = {v!r}")
                del normalized[k]
        return normalized

    def _parse_and_validate_tool_calls(
        self, tool_or_tools: Union[ToolCall, ToolChain, List[Dict[str, Any]]], instruction: str
    ) -> ToolChain:
        """
        Parses and validates one or more tool calls for execution.
        """
        # Normalize to a list of ToolCall for processing
        steps: List[ToolCall] = []
        # Accept raw dicts/lists for extra robustness
        if isinstance(tool_or_tools, ToolCall):
            steps = [tool_or_tools]
        elif isinstance(tool_or_tools, ToolChain):
            steps = tool_or_tools.steps
        elif isinstance(tool_or_tools, dict):
            steps = [ToolCall(**tool_or_tools)]
        elif isinstance(tool_or_tools, list):
            for item in tool_or_tools:
                if isinstance(item, ToolCall):
                    steps.append(item)
                elif isinstance(item, dict):
                    steps.append(ToolCall(**item))
                else:
                    logger.error(f"[Planner] Unknown item type in tool list: {type(item)}")
                    return self._fallback_llm_response(instruction)
        else:
            logger.error(
                f"[Planner] input is not ToolCall, ToolChain, dict, or list: got {type(tool_or_tools)}"
            )
            return self._fallback_llm_response(instruction)

        validated_tools: List[ToolCall] = []
        for i, step in enumerate(steps):
            tool_name = step.tool
            # 1) apply param alias normalization BEFORE validation
            params = self._normalize_param_aliases(step.params or {})

            # 2) Check tool exists
            if tool_name not in self.tool_registry.tools:
                logger.error(f"[Planner] Unknown tool at step {i}: '{tool_name}'. Fallback.")
                return self._fallback_llm_response(instruction)

            # 3) Filter/normalize params from tool registry (typed)
            valid_keys = self.tool_registry.get_param_keys(tool_name)
            logger.debug(f"[Planner] Param keys for '{tool_name}': {sorted(valid_keys)}")

            filtered_params = {k: v for k, v in params.items() if k in valid_keys}

            # 4) Compare to see if anything got dropped
            dropped = set(params) - set(filtered_params)
            if dropped:
                logger.info(
                    f"[Planner] Extra params filtered for tool '{tool_name}': {dropped}"
                )

            # 5) (Optional) light sanity: avoid obvious missing requireds causing crashes
            # If a tool expects 'path' and it's missing but instruction mentions a token like 'prompts', try to inject it.
            if "path" in valid_keys and "path" not in filtered_params:
                # naive extraction of a single folder/token word from the instruction
                m = re.search(r"\b(?:folder|dir(?:ectory)?)\s+([./\w\-]+)", instruction, re.IGNORECASE)
                if m:
                    filtered_params["path"] = m.group(1)
                    logger.info(f"[Planner] Filled missing 'path' from instruction: {filtered_params['path']!r}")

            validated_tools.append(ToolCall(tool=tool_name, params=filtered_params))

        if not validated_tools:
            logger.warning("[Planner] All tool calls invalid or filtered out. Fallback.")
            return self._fallback_llm_response(instruction)

        logger.info(f"[Planner] Returning ToolChain with {len(validated_tools)} validated tool call(s).")
        return ToolChain(steps=validated_tools)

    def _fallback_llm_response(self, instruction: str) -> ToolChain:
        """
        A generic catch-all fallback tool call.
        """
        fallback_call = ToolCall(tool="llm_response_async", params={"prompt": instruction})
        return ToolChain(steps=[fallback_call])

    # TODO: need to deprecate later - it should be handled by standard tool chain
    def _build_filtered_project_summary_plan(self) -> ToolChain:
        """
        Builds a multi-step agent plan to summarize the project using the
        list_project_files tool, read_files, aggregate_file_content, and
        llm_response_async tools.
        """
        plan: List[Dict[str, Any]] = []
        step_refs: List[str] = []

        # 1. List files using the tool (let tool handle exclusions and extensions)
        plan.append(
            {
                "tool": "list_project_files",
                "params": {
                    "root": ".",
                    "exclude": [".venv", "venv", "__pycache__", "model", "models"],
                    "include": [".py", ".md", ".toml", ".yaml", ".json"],
                },
            }
        )

        # 2. Read files from the output of list_project_files (limit to top 10)
        for i in range(10):
            plan.append(
                {
                    "tool": "read_files",
                    "params": {
                        "path": f"<step_0.files[{i}]>"
                    },
                }
            )
            step_refs.append(f"<step_{i+1}>")  # +1 because step_0 is list_project_files

        # 3. Aggregate file contents
        plan.append({"tool": "aggregate_file_content", "params": {"steps": step_refs}})

        # 4. Summarize with LLM
        prompts = load_summarizer_prompt()
        system_prompt = prompts.get("system_prompt", "")
        user_prompt = "\n".join(v for k, v in prompts.items() if k != "system_prompt" and v)

        # NOTE: llm_response_async input schema currently accepts 'prompt' only.
        plan.append(
            ToolCall(
                tool="llm_response_async",
                params={"prompt": f"{system_prompt}\n{user_prompt}".strip()},
            )
        )

        return ToolChain(steps=plan)

