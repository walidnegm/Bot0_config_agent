"""
agent/planner.py

Tool planning: Converts user instructions into a validated ToolChain (Pydantic) plan
using either a local or cloud LLM. Handles prompt construction, intent detection,
single vs multi-step planning, and plan validation.

- All public APIs always return ToolChain (never dict, list, or ToolCall).
"""

import re
import logging
from typing import Optional, Union, Type, Literal
import json
import asyncio
from pydantic import BaseModel
from tools.workbench.tool_registry import ToolRegistry

# from agent import llm_openai  # commented out: use a different api call function
# from agent.intent_classifier_core import classify_describe_only
from agent.intent_classifiers import (
    classify_describe_only_async,
    classify_task_decomposition_async,
)
from agent.llm_manager import get_llm_manager
from utils.llm_api_async import (
    call_openai_api_async,
    call_anthropic_api_async,
    # call_gemini_api_async,
)
from utils.prompt_logger import log_prompt_dict
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

logger = logging.getLogger(__name__)

OPENAI = "openai"  # Standardized to lowercase
ANTHROPIC = "anthropic"  # Standardized to lowercase
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
        response_model: Type[BaseModel],  # Default to text to be safe
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
            full_prompt = "\n".join(
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

    async def plan_async(self, instruction: str) -> ToolChain:
        """
        Asynchronously generate a plan (list of tool calls) from the user instruction.
        Uses the configured local or API LLM.

        Args:
            instruction (str): The user instruction.

        Returns:
            ToolChain: Validated, normalized sequence of tool calls.
            * Note: single tool call will be wrapped in multi-step model to be consistent.
        """
        instruction = instruction.strip()

        # todo: this should be pushed into standard multi-step later
        # todo: commented out to test the other intent classifier first
        # # 1. ✅ Intent classification - Describe Only
        # intent = await self._classify_intent_async(instruction)
        # if intent == "describe_project":
        #     return self._build_filtered_project_summary_plan()

        # 2. ✅ Intent classification - Task Decomposition (single or multi-step)
        task_decomp = await classify_task_decomposition_async(
            instruction, self
        )  # returns single or multi-step
        logger.info(f"[Planner] Task decomposition result: {task_decomp}")

        # 3. ✅ Load planner prompt templates (Jinja2 YAML-based)
        prompts_dic = load_planner_prompts(user_task=instruction)

        # 4. ✅ Set response type to JSON
        assert any(
            "return_json" in key for key in prompts_dic
        ), "Prompt error: must contain return json only."
        response_type = "json"

        # 5. ✅ Combine prompt templates based on single/multi step
        def combine_prompts(select_tool_promt: str) -> str:
            user_prompt = (
                select_tool_promt
                + "\n"
                + prompts_dic["return_json_only_prompt"]
                + "\n"
                + prompts_dic["user_task_prompt"]
            )
            return user_prompt

        system_prompt = prompts_dic.get("system_prompt", "")

        # 6. ✅ Call LLM
        # single_step: user prompt-> single tool & res model -> ToolCall
        if re.search(
            r"\bsingle[- ]?step\b", task_decomp, re.IGNORECASE
        ):  # use re to make it more "tight"
            user_prompt = combine_prompts(prompts_dic["select_single_tool_prompt"])

            # Log final planner prompt (human friendly)
            log_prompt_dict(
                logger=logger,
                label="Planner",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode="yaml",  # or "raw"/"json"
                level=logging.INFO,
            )

            result = await self.dispatch_llm_async(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                response_type=response_type,
                response_model=ToolCall,
            )
        else:  # multi-step: user prompt-> single tool & res model -> ToolCall
            user_prompt = combine_prompts(prompts_dic["select_multi_tool_prompt"])

            # Log final planner prompt (human-friendly)
            log_prompt_dict(
                logger=logger,
                label="Planner",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode="yaml",  # or "raw"/"json"
                level=logging.INFO,
            )

            result = await self.dispatch_llm_async(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                response_type=response_type,
                response_model=ToolChain,
            )  # return validated pydantic model

        # Extra safeguard
        if not isinstance(result, (ToolCall, ToolChain, dict, list)):
            logger.error(
                f"[Planner] LLM returned invalid plan type: {type(result)}. "
                "Falling back to default tool plan."
            )
            return self._fallback_llm_response(instruction)

        # 7. ✅ Parse Tool Calls (Validate, Normalize)
        tool_calls = self._parse_and_validate_tool_calls(result, instruction)

        # 8. ✅ Return final (validated, normalized) plan
        return tool_calls

    # todo: deprecate later; to_describe intent should be handled by standard tool chain
    async def _classify_intent_async(self, instruction: str) -> ToolChain | str:
        try:
            # Example: use_openai if API model is set
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

        Call the specified LLM API (OpenAI, Anthropic, or LLaMA3) with the provided prompt
        and return the response.

        Args:
            expected_res_type (Literal["json", "text", "code"], optional): Desired response format.
                Defaults to "text".
            response_model (Type[BaseModel], optional): Pydantic model used to validate and parse
                the LLM's output.
                Defaults to TextResponse.
        Returns:
            Union[JSONResponse, TabularResponse, TextResponse, CodeResponse, EditingResponseModel,
            JobSiteResponseModel]:
                The structured response from the API, validated if it passes JSON schema
                requirements.
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
        response_model: Type[BaseModel] = TextResponse,
    ) -> TextResponse | CodeResponse | JSONResponse | ToolCall | ToolChain:
        """
        Calls the local LLM with the provided prompt, using the specified response type and
        validation model.

        Runs the local LLM synchronously (via LLMManager), with optional system prompt, and
        validates output against the given Pydantic response model.

        Args:
            user_prompt (str): The user prompt (may include template expansion).
            system_prompt (Optional[str]): System prompt for the LLM (required by most local models).
            expected_res_type (Literal["json", "text", "code"], optional): Desired response format.
                Defaults to "text".
            response_model (Type[BaseModel], optional): Pydantic model used to validate and parse
                the LLM's output.
                Defaults to TextResponse.

        Returns:
            TextResponse | CodeResponse | JSONResponse | ToolCall | ToolChain:
                The validated output as a Pydantic model, matching the specified response_model.

        Notes:
            - The actual return type depends on response_model and LLM output.
            - For tool planning, response_model should be ToolCall or ToolChain.
            - May raise AssertionError if local_model_name is not set.
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
    ) -> ToolChain:
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
            ToolChain: A validated multi-step plan. If validation fails, returns
                a fallback ToolChain.
        """
        # Normalize to a list of ToolCall for processing
        steps = []
        # Accept raw dicts/lists for extra robustness
        if isinstance(tool_or_tools, ToolCall):
            steps = [tool_or_tools]
        elif isinstance(tool_or_tools, ToolChain):
            steps = tool_or_tools.steps
        elif isinstance(tool_or_tools, dict):
            steps = [ToolCall(**tool_or_tools)]
        elif isinstance(tool_or_tools, list):
            # Each item should be dict or ToolCall
            for item in tool_or_tools:
                if isinstance(item, ToolCall):
                    steps.append(item)
                elif isinstance(item, dict):
                    steps.append(ToolCall(**item))
                else:
                    logger.error(
                        f"[Planner] Unknown item type in tool list: {type(item)}"
                    )
                    return self._fallback_llm_response(instruction)
        else:
            logger.error(
                f"[Planner] input is not ToolCall, ToolChain, dict, or list: got {type(tool_or_tools)}"
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

            # Filter/normalize params from tool registry (typed)
            valid_keys = self.tool_registry.get_param_keys(tool_name)

            logger.debug(
                f"[Planner] Param keys for '{tool_name}': {sorted(valid_keys)}"
            )

            filtered_params = {k: v for k, v in params.items() if k in valid_keys}

            # Compare the two
            if len(filtered_params) != len(params):
                logger.info(
                    f"[Planner] Extra params filtered for tool '{tool_name}': "
                    f"{set(params) - valid_keys}"
                )

            validated_tools.append(ToolCall(tool=tool_name, params=filtered_params))

        if not validated_tools:
            logger.warning(
                "[Planner] All tool calls invalid or filtered out. Fallback."
            )
            return self._fallback_llm_response(instruction)

        logger.info(
            f"[Planner] Returning ToolChain with {len(validated_tools)} validated tool call(s)."
        )
        return ToolChain(steps=validated_tools)

    def _fallback_llm_response(self, instruction: str) -> ToolChain:
        """
        A generic catch-all fallback tool call.

        Always returns a ToolChain (pydantic model) with a single fallback step
        (even if a single step, wrap in List[ToolCall] / ToolChain.)
        """
        fallback_call = ToolCall(
            tool="llm_response_async", params={"prompt": instruction}
        )
        return ToolChain(steps=[fallback_call])

    # TODO: need to deprecate later - it should be handled by standard tool chain
    def _build_filtered_project_summary_plan(self) -> ToolChain:
        """
        Builds a multi-step agent plan to summarize the project using the
        list_project_files tool, read_files, aggregate_file_content, and
        llm_response_async tools.

        - Uses the list_project_files tool to gather relevant files (extensions,
            exclusions, size handled by tool).
        - Reads and aggregates files, then summarizes via LLM.

        Returns:
            ToolChain: The agent tool execution plan.
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
        for i in range(10):
            plan.append(
                {
                    "tool": "read_files",
                    "params": {
                        "path": f"<step_0.files[{i}]>"
                        # Indicates: take the i-th file from previous tool's 'files' output
                    },
                }
            )
            step_refs.append(f"<step_{i+1}>")  # +1 because step_0 is list_project_files

        # 3. Aggregate file contents
        plan.append({"tool": "aggregate_file_content", "params": {"steps": step_refs}})

        # 4. Summarize with LLM
        prompts = load_summarizer_prompt()
        system_prompt = prompts.get("system_prompt", "")
        user_prompt = "\n".join(
            v for k, v in prompts.items() if k != "system_prompt" and v
        )

        plan.append(
            ToolCall(
                tool="llm_response_async",
                params={"system_prompt": system_prompt, "user_prompt": user_prompt},
            )
        )

        return ToolChain(steps=plan)
