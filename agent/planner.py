"""
agent/planner.py
Tool planning: converts user instructions into a sequence of tool calls (plan), using either local or API LLMs.
Handles all prompt construction, intent detection, and result normalization.
"""

import json
import os
import re
import logging
import json
from typing import Dict, Any, List, Optional, Union, Type, Literal
from pydantic import BaseModel, ValidationError
import asyncio
from tools.tool_registry import ToolRegistry

# from agent import llm_openai  # commented out: use a different api call function
from agent.intent_classifier_core import classify_describe_only
from agent.llm_manager import get_llm_manager
from utils.llm_api_async import (
    call_openai_api_async,
    call_anthropic_api_async,
    # call_gemini_api_async,
)
from agent_models.llm_response_models import (
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

logger = logging.getLogger(__name__)

OPENAI = "openai"  # Standardized to lowercase
ANTHROPIC = "anthropic"  # Standardized to lowercase
# class ToolCall(BaseModel):
#     tool: str
#     params: Dict[str, Any]
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
        elif self.local_model_name:
            logger.info(f"[Planner] ⚙️ Using local LLM: {self.local_model_name}")
        else:
            logger.error("[Planner] No LLM model specified!")
            raise ValueError("Must specify either local_model_name or api_model_name.")

        self.param_aliases = {
            "file_path": "path",
            "filename": "path",
            "filepath": "path",
        }

    async def dispatch_llm_async(
        self,
        user_prompt: str,
        system_prompt: str,
        response_type: Literal["json", "text", "code"],
        response_model: Optional[Type[BaseModel]] = None,
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

        Args:
            user_prompt (str): User input text prompt with auto-generated template.
            system_prompt (str): System prompt template (required by LOCAL LLMs).
            response_type (str, optional): Desired response format (e.g., "text", "json",
                "code"). Defaults to "text".
            **kwargs: Additional keyword arguments for the LLM call (e.g., temperature,
                max_tokens, response_model).

        Returns:
            Union[dict, JSONResponse, TextResponse, CodeResponse, ToolCall, ToolChain]:
                The LLM's response as a structured Pydantic model or a plain dict,
                depending on provider and settings.

        Raises:
            ValueError: If neither a local nor API model is configured.

        Notes:
            - Cloud LLMs (OpenAI, Anthropic, etc.) are always called asynchronously.
            - Local LLMs are called in a thread pool to avoid blocking the event loop.
            - You need to post-process the result (e.g., call .model_dump()
                if you always want a dict).
        """
        # API LLMs: use async (faster)
        if self.api_model_name:
            full_prompt = "/n".join(
                [system_prompt, user_prompt]
            )  # Combine use & sys prompts
            result = await self._call_api_llm_async(
                prompt=full_prompt,
                expected_res_type=response_type,
                response_model=response_model,
            )
            return result  # return pydantic model

        # Local LLMs: run sync (VRAM constraint) but use loop.run_in_executor to keep
        # the event loop non-blocking
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
            return result  # return pydantic model
        else:
            raise ValueError(
                f"Unknown model name: {self.api_model_name or self.local_model_name}"
            )

    async def plan_async(self, instruction: str) -> List[Dict[str, Any]]:
        """
        Asynchronously generate a plan (list of tool calls) from the user instruction.
        Uses the configured local or API LLM.

        Args:
            instruction (str): The user instruction.

        Returns:
            List[Dict[str, Any]]: Sequence of planned tool call dicts.
        """
        instruction = instruction.strip()

        # 1. ✅ Intent Detection
        intent = self._classify_intent(instruction)
        if intent == "describe_project":
            return self._build_filtered_project_summary_plan()

        # 2. ✅ Prompt Construction (Jinja2 YAML-based)
        prompts_dic = self._load_prompt_segments(instruction)

        # Infer response type and json_type
        response_type = "text"
        response_model = None

        if any("return_json" in key for key in prompts_dic.keys()):
            response_type = "json"
        # Use response_model->call_llm functionsmodels->determine how to validate response
        if any("tools" in key for key in prompts_dic.keys()):
            response_model = ToolChain | ToolChain

        # Combine prompts
        system_prompt = prompts_dic.get("system_prompt", "")
        user_prompt = "\n".join(
            v for k, v in prompts_dic.items() if k != "system_prompt" and v
        )  # combine all non-system and empty prompts

        # Call LLM to select tool(s)
        result = await self.dispatch_llm_async(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_type=response_type,
            response_model=response_model,
        )  # return validated pydantic model

        # 4. Parse Tool Calls (Validate, Normalize)
        tool_calls = self._parse_and_validate_tool_calls(result, instruction)

        # 5. Return final (validated, normalized) plan
        return tool_calls

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
        expected_res_type: Literal["json", "text", "code"] = "text",
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Union[
        JSONResponse,
        TextResponse,
        CodeResponse,
        ToolCall,
        ToolChain,
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

        # Set api llm parameters
        llm_provider = get_llm_provider(self.api_model_name)
        model_id = self.api_model_name
        max_tokens = API_LLM_MAX_TOKEN_DEFAULT
        temperature = API_LLM_TEMPERATURE_DEFAULT

        # Choose the appropriate LLM API
        if llm_provider.lower() == "openai":
            validated_result = await call_openai_api_async(
                prompt=prompt,
                model_id=model_id,
                expected_res_type=expected_res_type,
                temperature=temperature,
                max_tokens=max_tokens,
                response_model=response_model,
                # client=client,  # todo: commented out for now, but it's good to instantiate early for cloud APIs
            )
        elif llm_provider.lower() == "anthropic":
            validated_result = await call_anthropic_api_async(
                prompt=prompt,
                model_id=model_id,
                expected_res_type=expected_res_type,
                temperature=temperature,
                max_tokens=max_tokens,
                response_model=response_model,
                # client=self.client, # todo: commented out for now, but it's good to instantiate early for cloud APIs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        return validated_result

    def _call_local_llm(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        expected_res_type: Literal["json", "text", "code"] = "text",
        response_model: Optional[Type[BaseModel]] = None,
    ) -> TextResponse | CodeResponse | JSONResponse | ToolCall | ToolChain:
        """
        Call the local LLM with the given prompt and parameters. Use cache to save VRAM

        Args:
            prompt (str): The formatted prompt.
            model_name (str): Which local model to use.
            ...

        Returns:
            JSONResponse (or your response model)
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

    def _parse_and_validate_tool_calls(
        self, tool_or_tools: Union[ToolCall, ToolChain], instruction: str
    ) -> List[dict]:
        """
        Parses and validates one or more tool calls for execution.

        Accepts either a single ToolCall (representing a one-step tool plan)
        or a ToolChain (multi-step plan containing a list of ToolCalls), both as
        Pydantic models.

        For each tool call:
            - Verifies that the tool name exists in the current tool registry.
            - Filters parameters to include only those defined in the registry schema.
            - Logs and falls back to a default response if any tool is unknown or if no
                valid steps remain.

        Args:
            tool_or_tools (Union[ToolCall, ToolChain]):
                A ToolCall instance (for a single-step plan) or a ToolChain instance
                (for a multi-step plan).
            instruction (str):
                The original user instruction, used for logging and fallback response
                    generation.

        Returns:
            List[dict]: A validated list of tool call dictionaries, each containing 'tool'
                        and 'params' keys, suitable for passing to the executor.
                        If validation fails, returns a fallback plan.
        """

        # Normalize to a list of ToolCall for processing
        if isinstance(tool_or_tools, ToolCall):
            steps = [tool_or_tools]
        elif isinstance(tool_or_tools, ToolChain):
            steps = tool_or_tools.steps
        else:
            logger.error(
                f"[Planner] input is not ToolCall or ToolChain: got {type(tool_or_tools)}"
            )
            return self._fallback_llm_response(instruction)

        validated_tools = []
        for i, step in enumerate(steps):
            tool_name = step.tool
            params = step.params

            # Check tool exists
            if tool_name not in self.tool_registry.tools:
                logger.error(
                    f"[Planner] Unknown tool at step {i}: '{tool_name}'. Fallback."
                )
                return self._fallback_llm_response(instruction)

            # Filter/normalize params for registry
            valid_keys = set(
                self.tool_registry.tools[tool_name]
                .get("parameters", {})
                .get("properties", {})
                .keys()
            )
            filtered_params = {k: v for k, v in params.items() if k in valid_keys}
            if len(filtered_params) != len(params):
                logger.info(
                    f"[Planner] Extra params filtered for tool '{tool_name}': {set(params) - valid_keys}"
                )

            validated_tools.append(
                {
                    "tool": tool_name,
                    "params": filtered_params,
                }
            )

        if not validated_tools:
            logger.warning(
                "[Planner] All tool calls invalid or filtered out. Fallback."
            )
            return self._fallback_llm_response(instruction)

        logger.info(
            f"[Planner] Returning {len(validated_tools)} validated tool call(s)."
        )
        return validated_tools

    def _fallback_llm_response(self, instruction: str) -> List[Dict[str, Any]]:
        return [{"tool": "llm_response_async", "params": {"prompt": instruction}}]

    # --- Utility: Parameter Normalization, Placeholder Rewrites ---
    def _normalize_params(self, params: Dict) -> Dict:
        """
        Normalizes and sanitizes a single tool call's parameter dictionary.

        - Applies any parameter aliasing (e.g., converts 'filename' or 'filepath' to 'path').
        - Detects placeholder path values such as 'path/to/file', 'your_file',
            or strings containing 'placeholder'.
            If a placeholder is detected in the 'path' parameter, rewrites parameters to
            suggest a keyword-based file search instead.
        - Returns the cleaned parameters dict, or a fallback dict for file search
            if a placeholder is found.

        Args:
            params (dict): Parameter dictionary for a single tool call.

        Returns:
            dict: Normalized and sanitized parameters, ready for tool execution.
        """
        params = {self.param_aliases.get(k, k): v for k, v in params.items()}
        placeholder_path = params.get("path", "")
        if any(
            kw in str(placeholder_path) for kw in ["path/to", "your_", "placeholder"]
        ):
            logger.warning(f"Placeholder path '{placeholder_path}' detected.")
            return {"keywords": ["python", "py"], "root": "."}
        return params

    def _build_filtered_project_summary_plan(self) -> List[Dict[str, Any]]:
        """
        Builds a multi-step agent plan to summarize the project using the
        list_project_files tool, read_file, aggregate_file_content, and
        llm_response_async tools.

        - Uses the list_project_files tool to gather relevant files (extensions,
            exclusions, size handled by tool).
        - Reads and aggregates files, then summarizes via LLM.

        Returns:
            List[Dict[str, Any]]: The agent tool execution plan.
        """
        plan = []
        step_refs = []

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

        # 2. Read files from the output of list_project_files (limit to top 5)
        # - "<step_0>" is the step reference for list_project_files
        # - Agent/executor should use output from step_0 to dynamically generate these steps
        for i in range(5):
            plan.append(
                {
                    "tool": "read_file",
                    "params": {
                        "path": f"<step_0.files[{i}]>"
                        # Indicates: take the i-th file from previous tool's 'files' output
                    },
                }
            )
            step_refs.append(f"<step_{i+1}>")  # +1 because step_0 is list_project_files

        # 3. Aggregate file contents
        plan.append({"tool": "aggregate_file_content", "params": {"steps": step_refs}})

        # 4. Load summarizer prompt templates
        prompts = load_summarizer_prompt()
        system_prompt = prompts.get("system_prompt", "")
        user_prompt = "\n".join(
            v for k, v in prompts.items() if k != "system_prompt" and v
        )

        # 5. Summarize with LLM
        plan.append(
            {
                "tool": "llm_response_async",
                "params": {"system_prompt": system_prompt, "user_prompt": user_prompt},
            }
        )

        return plan
