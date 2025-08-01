"""
agent/planner.py
Tool planning: converts user instructions into a sequence of tool calls (plan), using either local or API LLMs.
Handles all prompt construction, intent detection, and result normalization.
"""

import json
import os
import re
import logging
from typing import Dict, Any, List, Optional, Union, Type
from pydantic import BaseModel, ValidationError
import asyncio
import concurrent.futures
from tools.tool_registry import ToolRegistry

# from agent import llm_openai  # commented out: use a different api call function
from agent.intent_classifier_core import classify_describe_only
from agent.llm_manager import (
    LLMManager,
)
from utils.llm_api_async import (
    call_openai_api_async,
    call_anthropic_api_async,
    # call_gemini_api_async,
)
from agent_models.llm_response_models import (
    JSONResponse,
    TextResponse,
    CodeResponse,
    ToolSelect,
    ToolSteps,
)
from prompts.load_agent_prompts import load_planner_prompts
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

logger = logging.getLogger(__name__)

OPENAI = "openai"  # Standardized to lowercase
ANTHROPIC = "anthropic"  # Standardized to lowercase
# class ToolCall(BaseModel):
#     tool: str
#     params: Dict[str, Any]


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
    Planner: Handles tool planning using either a local LLM or a cloud API LLM (async-safe).
    All prompt segments are rendered from Jinja2 YAML templates.
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
            self.llm_manager = None
        elif self.local_model_name:
            logger.info(f"[Planner] ⚙️ Using local LLM: {self.local_model_name}")
            self.llm_manager = LLMManager(model_name=self.local_model_name)
        else:
            logger.error("[Planner] No LLM model specified!")
            raise ValueError("Must specify either local_model_name or api_model_name.")

        self.param_aliases = {
            "file_path": "path",
            "filename": "path",
            "filepath": "path",
        }

        # --- Intent Detection ---

    def _classify_intent(self, instruction: str) -> str:
        try:
            # Example: use_openai if API model is set
            return classify_describe_only(
                instruction, use_openai=bool(self.api_model_name)
            )
        except Exception as e:
            logger.warning(f"Intent classifier failed, defaulting to LLM. Error: {e}")
            return "unknown"

    # --- Prompt Construction ---
    def _load_prompt_segments(self, instruction: str) -> Dict[str, Any]:
        tools_for_prompt = [
            {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters", {}).get("properties", {}),
                "usage_hint": spec.get("usage_hint", ""),
            }
            for name, spec in self.tool_registry.get_all().items()
        ]
        # Render prompt segments via the standard file YAML/Jinja2 file (yaml.j2)
        return load_planner_prompts(user_task=instruction, tools=tools_for_prompt)

    async def _call_api_llm_async(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        expected_res_type:str="text",
        response_model: Optional[Type[BaseModel]]=None
    ) -> Union[
        JSONResponse,
        TextResponse,
        CodeResponse,
        ToolSelect,
        ToolSteps,
    ]:
        """
        *Async version of the method.

        Call the specified LLM API (OpenAI, Anthropic, or LLaMA3) with the provided prompt
        and return the response.

        Args:
            - prompt (str): The formatted prompt to send to the LLM API.
            - temperature (float, optional): Temperature setting for this specific API call.
                                           If None, uses the class-level temperature.

        Returns:
            Union[JSONResponse, TabularResponse, TextResponse, CodeResponse, EditingResponseModel,
            JobSiteResponseModel]:
                The structured response from the API, validated if it passes JSON schema requirements.
        """
        # Get llm provider (openai, anthropic, gemini, etc.)
        assert (
            self.api_model_name
        ), "No API model specified. Please select a model before proceeding."
        validate_api_model_name(
            self.api_model_name
        )  # Raises error if model names do not matach internal config
        llm_provider = get_llm_provider(self.api_model_name)
        model_id = self.api_model_name

        # Choose the appropriate LLM API
        if llm_provider.lower() == "openai":
            response_model = await call_openai_api_async(
                prompt=prompt,
                model_id=model_id,
                expected_res_type=expected_res_type,
                json_type="tool_selection",
                temperature=temperature,
                max_tokens=max_tokens,
                # client=client,  # todo: commented out for now, but it's good to instantiate early for cloud APIs
            )
        elif llm_provider.lower() == "anthropic":
            response_model = await call_anthropic_api_async(
                prompt=prompt,
                model_id=model_id,
                expected_res_type="json",
                json_type="tool_selection",
                temperature=temperature,
                max_tokens=max_tokens,
                # client=self.client, # todo: commented out for now, but it's good to instantiate early for cloud APIs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        return response_model

    def _call_local_llm(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        expected_res_type: str="text",
        json_type: Optional[str] = None

    ) -> TextResponse | CodeResponse | JSONResponse | ToolSelect | ToolSteps:
        """
        Call the local LLM with the given prompt and parameters.

        Args:
            prompt (str): The formatted prompt.
            model_name (str): Which local model to use.
            ...

        Returns:
            JSONResponse (or your response model)
        """

        llm = LLMManager(model_name=model_name or self.local_model_name)
        result = llm.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            expected_res_type="text",  # or whatever fits your workflow
        )
        # You might want to post-process/validate as a response model
        return result  # ideally as a Pydantic model, not just a string!

    async def dispatch_llm_async(
        self,
        prompt: str,
        response_type: str = "text", # default to text
        json_type: Optional[str]=None, 
        **kwargs,
    ) -> Union[
        dict,
        JSONResponse,
        TextResponse,
        CodeResponse,
        ToolSelect,
        ToolSteps,
    ]:
        """
        Asynchronously route the LLM call to either a cloud API (async) or a local model
        (run in executor).

        Args:
            prompt (str): The formatted prompt for the LLM.
            **kwargs: Additional keyword arguments passed to the LLM call.

        Returns:
            Union[dict, JSONResponse, TextResponse, CodeResponse, ToolSelect, ToolSteps]:
                The structured response from the LLM, as a dict or Pydantic model.
        """
        if self.api_model_name:
            result = await self._call_api_llm_async(prompt=prompt, **kwargs)
            return (
                result  # Could call _maybe_model_dump(result) if you want always dict
            )
        elif self.local_model_name:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._call_local_llm(prompt=prompt, **kwargs)
            )
            return (
                result  # Could call _maybe_model_dump(result) if you want always dict
            )
        else:
            raise ValueError(
                f"Unknown model name: {self.api_model_name or self.local_model_name}"
            )

    def _call_llm(self, prompts: dict) -> str:
        # Handles async/sync, local/api dispatch (stub: real implementation depends on your infra)
        system_prompt = prompts.get("system_prompt")
        full_prompt = "\n".join(prompts.values())
        # If using an injected backend, just call:
        # return self.llm_planner.generate_plan(full_prompt, system_prompt, ...)
        if self.api_model_name:
            # Call your async API LLM (pseudo-code: use actual async/await if needed)
            from utils.llm_api_async import (
                call_openai_api_async,
                call_anthropic_api_async,
            )

    def plan(self, instruction: str) -> List[Dict[str, Any]]:
        """
        Asynchronously generate a plan (list of tool calls) from the user instruction.
        Uses the configured local or API LLM.

        Args:
            instruction (str): The user instruction.

        Returns:
            List[Dict[str, Any]]: Sequence of planned tool call dicts.
        """
        instruction = instruction.strip()

        # 1. Intent Detection
        intent = self._classify_intent(instruction)
        if intent == "describe_project":
            return self._summary_plan()

        # 2. Prompt Construction (Jinja2 YAML-based)
        prompts = self._load_prompt_segments(instruction)

        # Infer response type and json_type
        if any("return_json" in key for key in prompts.keys()):
            response_type = "json"
        if any("tools" in key for key in prompts.keys()): 
            json_type = "tool_selection"
            response_model = ToolSteps|ToolSteps

        result = await self._call_api_llm_async(
            prompt=full_prompt,
            ...,
            response_model=response_model,
        )

        # 3. LLM Routing (Local vs. API)
        llm_output = self._dispatch_llm(prompts)

        # 4. Parse Tool Calls (Validate, Normalize)
        tool_calls = self._parse_and_validate_tool_calls(llm_output, instruction)

        # 5. Return final (validated, normalized) plan
        return tool_calls

        # ✅ Run intent classification before using the LLM's tool plan
        try:
            intent = classify_describe_only(
                instruction, use_openai=bool(self.api_model_name)
            )

        except Exception as e:
            logger.warning(
                f"Intent classifier failed, defaulting to normal LLM plan. Error: {e}"
            )
            intent = "unknown"

        logger.info(f"[Planner] 🧠 Parsed intent: {intent}")
        if intent == "describe_project":
            logger.info(
                "[Planner] 🔁 Overriding tool plan — injecting describe_project summary plan"
            )
            return self._build_filtered_project_summary_plan()

        # Load all prompt segments from template
        prompts = self._load_prompts(instruction)
        system_prompt = prompts["system_prompt"]
        full_prompt = (
            prompts["select_tools_prompt"]
            + "\n"
            + prompts["return_json_only_prompt"]
            + "\n"
            + prompts["multi_step_prompt"]
            + "\n"
            + prompts["user_prompt"]
        )

        logger.debug("\n[Planner] 📜 System prompt:\n%s", system_prompt)
        logger.debug("\n[Planner] 📜 Full prompt:\n%s", full_prompt)

        # --- Major Change: Route between async API and sync local LLM ---
        if self.api_model_name:
            logger.info(f"[Planner] Calling API LLM: {self.api_model_name}")
            # Example routing (update for your API models!):
            if "gpt" in self.api_model_name or "o3" in self.api_model_name:
                llm_output = await call_openai_api_async(
                    prompt=full_prompt,
                    model=self.api_model_name,
                    system_prompt=system_prompt,
                    max_tokens=512,
                    temperature=0.0,
                )
            elif "claude" in self.api_model_name:
                llm_output = await call_anthropic_api_async(
                    prompt=full_prompt,
                    model=self.api_model_name,
                    system_prompt=system_prompt,
                    max_tokens=512,
                    temperature=0.0,
                )
            elif "gemini" in self.api_model_name:
                llm_output = await call_gemini_api_async(
                    prompt=full_prompt,
                    model=self.api_model_name,
                    system_prompt=system_prompt,
                    max_tokens=512,
                    temperature=0.0,
                )
            else:
                logger.error(f"Unknown API model for async call: {self.api_model_name}")
                raise ValueError(f"Unknown API model: {self.api_model_name}")
        else:
            loop = asyncio.get_event_loop()

            def blocking_generate():
                return self.llm_manager.generate(
                    full_prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=512,
                    temperature=0.0,
                )

            llm_output = await loop.run_in_executor(None, blocking_generate)

        if isinstance(llm_output, dict):
            llm_output = llm_output.get("text") or llm_output.get("output") or ""

        logger.debug("\n[Planner] 📤 LLM raw response:\n%s", repr(llm_output))

        # --- (Rest: plan extraction/validation unchanged except for clean logging) ---
        try:
            extracted_json = llm_output.strip()
            logger.debug("\n[Planner] ✅ Extracted JSON array:\n%s", extracted_json)

            raw_tool_calls = json.loads(extracted_json)
            validated_calls: List[ToolSelect] = []

            for i, item in enumerate(raw_tool_calls):
                tool_name = item.get("tool")
                params = item.get("params", {})
                # 🔧 Normalize parameter keys
                for old_key, new_key in self.param_aliases.items():
                    if old_key in params and new_key not in params:
                        params[new_key] = params.pop(old_key)
                # 🔧 Remove unexpected params
                if tool_name not in self.tool_registry.tools:
                    logger.warning(f"[Planner] ⚠️ Unknown tool: {tool_name}")
                    return [{"tool": "llm_response", "params": {"prompt": instruction}}]

                # Autofix: 'files' param to 'root'
                if (
                    tool_name == "list_project_files"
                    and "files" in params
                    and "root" not in params
                ):
                    logger.warning("[Planner] ⚠️ Auto-rewriting 'files' param to 'root'")
                    params["root"] = "."
                    del params["files"]

                # Placeholder handling
                placeholder_path = params.get("path", "")
                if any(
                    kw in str(placeholder_path)
                    for kw in ["path/to", "your_", "placeholder"]
                ):
                    logger.warning(
                        f"[Planner] ⚠️ Placeholder path '{placeholder_path}' detected."
                    )
                    return [
                        {
                            "tool": "find_file_by_keyword",
                            "params": {"keywords": ["python"], "root": "."},
                        },
                        {
                            "tool": "echo_message",
                            "params": {"message": "<prev_output>"},
                        },
                    ]

                if any(
                    "path/to/" in str(v) or "your/" in str(v) for v in params.values()
                ):
                    logger.warning(
                        f"[Planner] ⚠️ Placeholder detected → rewriting to find_file_by_keyword + echo_message."
                    )
                    return [
                        {
                            "tool": "find_file_by_keyword",
                            "params": {"keywords": ["python", "py"], "root": "."},
                        },
                        {
                            "tool": "echo_message",
                            "params": {"message": "<prev_output>"},
                        },
                    ]

                valid_keys = set(
                    self.tool_registry.tools[tool_name]
                    .get("parameters", {})
                    .get("properties", {})
                    .keys()
                )
                params = {k: v for k, v in params.items() if k in valid_keys}
                item["params"] = params
                logger.debug(f"[Planner] 🔄 Normalized call {i}: {item}")

                try:
                    validated = ToolSelect(**item)
                    if validated.tool not in self.tool_registry.tools:
                        logger.warning(
                            f"[Planner] ⚠️ Invalid tool: {validated.tool}. Falling back to llm_response."
                        )
                        return [
                            {"tool": "llm_response", "params": {"prompt": instruction}}
                        ]
                    validated_calls.append(validated)
                except ValidationError as ve:
                    logger.error(f"[Planner] ❌ Validation error in item {i}:\n{ve}\n")
                    return [{"tool": "llm_response", "params": {"prompt": instruction}}]

            if extracted_json.strip() == "[]" or not validated_calls:
                logger.info("[Planner] 🤷 No valid tools matched. Using llm_response.")
                return [{"tool": "llm_response", "params": {"prompt": instruction}}]

            logger.info("\n[Planner] 🔍 Final planned tools:")
            for call in validated_calls:
                logger.info(f"  → {call.tool} with params {call.params}")

            return [call.dict() for call in validated_calls]

        except Exception as e:
            logger.error(
                f"[Planner] ❌ Failed to parse tools JSON: {e}\n", exc_info=True
            )
            return [{"tool": "llm_response", "params": {"prompt": instruction}}]

    def _build_filtered_project_summary_plan(self) -> List[Dict[str, Any]]:
        files_to_read = []
        for root, _, files in os.walk("."):
            if any(skip in root for skip in ["venv", "__pycache__", "models"]):
                continue
            for fname in files:
                if not fname.endswith((".py", ".md", ".toml")):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    if os.path.getsize(fpath) <= 10_000:
                        files_to_read.append(fpath)
                except OSError:
                    continue

        files_to_read = sorted(files_to_read)[:5]

        plan = []
        step_refs = []
        for idx, fpath in enumerate(files_to_read):
            plan.append({"tool": "read_file", "params": {"path": fpath}})
            step_refs.append(f"<step_{idx}>")

        plan.append({"tool": "aggregate_file_content", "params": {"steps": step_refs}})
        plan.append(
            {
                "tool": "llm_response",
                "params": {
                    "prompt": (
                        "Give a concise summary of the project based on the following files:\n\n"
                        "<prev_output>\n\nHighlight purpose, key components, and usage."
                    )
                },
            }
        )

        return plan
