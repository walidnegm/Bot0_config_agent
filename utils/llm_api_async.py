"""Async version of llm_api_utils_async.py

This module provides asynchronous utility functions for interacting with various LLM APIs,
including OpenAI, Anthropic, and Llama3. It handles API calls, validates responses,
and manages provider-specific nuances such as single-block versus multi-block responses.

Key Features:
# * - API clients are default to global to control for rate limit based on
# * LLM provider specific rate limits to keep from overloading!
- Asynchronous support for OpenAI and Anthropic APIs.
- Compatibility with synchronous Llama3 API via an async executor.
- Validation and structuring of responses into Pydantic models.
- Modular design to accommodate provider-specific response handling.

Modules and Methods:
- `call_openai_api_async`: Asynchronously interacts with the OpenAI API.
- `call_anthropic_api_async`: Asynchronously interacts with the Anthropic API.
- `call_llama3_async`: Asynchronously interacts with the Llama3 API using a synchronous executor.
- `call_api_async`: Unified async function for handling API calls with validation.
#* - `run_in_executor_async`: Executes synchronous functions in an async context.
#* (Optional - for future use)
- Validation utilities (e.g., `validate_response_type`, `validate_json_type`).

Usage:
This module is intended for applications that require efficient and modular integration
with multiple LLM providers.
"""

# todo: need to later add gemini and/or grok api call functions...


# Built-in & External libraries
import asyncio
from typing import cast, Optional, Type, Tuple, Union
import json
from random import uniform
import logging
from pydantic import BaseModel, ValidationError
import httpx
from aiolimiter import AsyncLimiter

# LLM imports
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from anthropic._exceptions import RateLimitError

# Project level imports
from agent_models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TextResponse,
    ToolCall,
    ToolChain,
)
from utils.get_llm_api_keys import (
    get_anthropic_api_key,
    get_openai_api_key,
)
from agent_models.llm_response_validators import (
    validate_response_type,
    validate_tool_selection_or_steps,
)
from configs.api_models import (
    GPT_4_1,
    GPT_4_1_MINI,
    GPT_4_1_NANO,
    GPT_4O,
    CLAUDE_HAIKU,
    CLAUDE_SONNET_4,
    CLAUDE_OPUS,
)

logger = logging.getLogger(__name__)

# llm_providers
OPENAI = "openai"
ANTHROPIC = "anthropic"
GEMINI = "gemini"


# Global clients (instantiated once at module load)
OPENAI_CLIENT = AsyncOpenAI(api_key=get_openai_api_key(), timeout=httpx.Timeout(10.0))
ANTHROPIC_CLIENT = AsyncAnthropic(
    api_key=get_anthropic_api_key(), timeout=httpx.Timeout(10.0)
)

# Provider-specific rate limiters (requests per minute)
# Adjust based on your OpenAI tier and Anthropic plan
RATE_LIMITERS = {
    OPENAI: AsyncLimiter(max_rate=1000, time_period=60),
    ANTHROPIC: AsyncLimiter(
        max_rate=50, time_period=60
    ),  # e.g., Anthropic free tier: 50 RPM
}


async def with_rate_limit_and_retry(api_func, llm_provider: str):
    """Apply rate limiting and retries with exponential backoff."""
    async with RATE_LIMITERS[llm_provider]:
        max_retries = 5
        base_delay = 1
        for attempt in range(max_retries):
            try:
                return await api_func()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 529] and attempt < max_retries - 1:
                    retry_after = int(e.response.headers.get("Retry-After", base_delay))
                    wait_time = min(
                        10, retry_after + (attempt * base_delay) + uniform(0, 1)
                    )
                    logger.warning(
                        f"Rate limit hit for {llm_provider}, retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except httpx.TimeoutException as e:
                logger.error(f"Timeout for {llm_provider}: {e}")
                raise


# Unified async API calling function
async def call_api_async(
    client: Optional[Union[AsyncOpenAI, AsyncAnthropic]],
    model_id: str,
    prompt: str,
    expected_res_type: str,
    temperature: float,
    max_tokens: int,
    llm_provider: str,
    response_model: Optional[Type[BaseModel] | Tuple[Type[BaseModel], ...]] = None,
) -> Union[JSONResponse, CodeResponse, TextResponse, ToolCall, ToolChain]:
    """
    Unified async LLM API calling and response validation function.

    Makes a single API call to an LLM provider (OpenAI or Anthropic),
    applies standard rate limiting and retry logic, and validates the
    response using project-specific structured Pydantic models.

    Key features:
    - Handles provider-specific API differences (e.g., OpenAI vs Anthropic).
    - Optionally validates tool-calling JSON against stricter internal models.
    - Centralizes error handling, logging, and retry logic.
    - Returns a fully validated structured response model for downstream use.

    Args:
        client: The async client instance for the selected LLM provider.
                Must be an AsyncOpenAI or AsyncAnthropic instance (as appropriate),
                or None if instantiating internally.
        model_id: Model identifier (e.g., "gpt-4-turbo" or "claude-3-opus").
        prompt: The user or system prompt to send to the model.
        expected_res_type: The desired output type ("json", "text", "code", etc.).
        temperature: Sampling temperature for LLM output.
        max_tokens: Maximum tokens to generate in the response.
        llm_provider: The LLM provider key ("openai", "anthropic").
        response_model: (Optional) Pydantic model to validate response
            (e.g., ToolChain).

    Returns:
        One of the project's structured response types (JSONResponse, CodeResponse,
        TextResponse, ToolCall, ToolChain), validated and ready for further use.

    Raises:
        ValueError: For invalid, empty, or non-parseable API responses, or
            failed validation.
        TypeError: If the response content does not match the expected response type.
        NotImplementedError: If an unsupported provider is specified.
        Exception: For all other unexpected errors (e.g., network, rate
            limit failures).

    Notes:
        - Only "openai" and "anthropic" providers are currently supported.
        - If additional providers (e.g., Llama.cpp API) are needed, add a new branch.
        - The response is validated twice: (1) for top-level type (e.g., JSONResponse),
            (2) for subtype-specific details if expected_res_type is "json" and
            json_type is recognized.
        - Rate limiting and retry logic is handled automatically using
            with_rate_limit_and_retry() for robustness in production environments.
        - OpenAI & Llama3 always returns single-block responses, while anthropic may
            return multi-block responses, which needs special treatment.
        #*  Therefore, the API calling for each LLM provider need to remain separate:
            #* Combining them into a single code block will have complications;
            #* keep them separate here for each provider is a more clean and modular.

    Example:
        >>> await call_api_async(
                client=openai_client,
                model_id="gpt-4-turbo",
                prompt="List the tools in this project.",
                expected_res_type="json",
                temperature=0.2,
                max_tokens=512,
                llm_provider="openai"
                response_model=ToolChain
            )
    """
    try:
        logger.info(f"Making API call with expected response type: {expected_res_type}")
        response_content = ""

        if llm_provider.lower() == OPENAI:
            openai_client = cast(AsyncOpenAI, client)

            async def openai_request():
                return await openai_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            response = await with_rate_limit_and_retry(openai_request, llm_provider)
            if not response or not response.choices:
                raise ValueError("OpenAI API returned an invalid or empty response.")
            response_content = response.choices[0].message.content

        elif llm_provider.lower() == ANTHROPIC:
            anthropic_client = cast(AsyncAnthropic, client)
            system_instruction = (
                "You are a helpful assistant who adheres to instructions."
            )

            async def anthropic_request():
                return await anthropic_client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": system_instruction + prompt}],
                    temperature=temperature,
                )

            response = await with_rate_limit_and_retry(anthropic_request, llm_provider)
            if not response or not response.content:
                raise ValueError("Empty response received from Anthropic API")
            first_block = response.content[0]
            response_content = (
                first_block.text if hasattr(first_block, "text") else str(first_block)
            )
            if not response_content:
                raise ValueError("Empty content in response from Anthropic API")

        logger.info(f"Raw {llm_provider} Response: {response_content}")

        # Validation 1: response content and return structured response
        validated_response_model = validate_response_type(
            response_content, expected_res_type
        )
        logger.info(
            f"validated response content after validate_response_type: \n{validated_response_model}"
        )  # TODO: debugging; delete afterwards

        # Validation 2: custom logic for specific tool JSON types
        if expected_res_type == "json":
            if not isinstance(validated_response_model, JSONResponse):
                raise TypeError(
                    "Expected a JSONResponse model when response type is 'json'."
                )

            if response_model:
                # Normalize to tuple for flexible membership testing
                if not isinstance(response_model, tuple):
                    response_models = (response_model,)
                else:
                    response_models = response_model

                # Custom logic: If any response_model is ToolCall or ToolChain
                if any(m in (ToolCall, ToolChain) for m in response_models):
                    response_data = validated_response_model.data
                    try:
                        validated_response_model = validate_tool_selection_or_steps(
                            response_data
                        )
                    except ValidationError as ve:
                        logger.error("Tool selection validation failed: %s", ve)
                        raise ValueError(
                            f"Tool selection validation failed: {ve}"
                        ) from ve

        logger.info(
            f"validated response content after validate with response model: \n{validated_response_model}"
        )  # TODO: debugging; delete afterwards
        return validated_response_model

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from {llm_provider} API: {e}")
    except Exception as e:
        logger.error(f"{llm_provider} API call failed: {e}")
        raise


# Async wrapper for OpenAI
async def call_openai_api_async(
    prompt: str,
    model_id: str = GPT_4_1_MINI,  # default to 4.1 mini
    expected_res_type: str = "str",
    # json_type: str = "tool_selection",  # default to tool_selection for now
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[AsyncOpenAI] = None,  # Default to global client
    response_model: Optional[Type[BaseModel]] = None,
) -> Union[JSONResponse, TextResponse, CodeResponse, ToolCall, ToolChain]:
    """
    Asynchronous convenience wrapper for calling the OpenAI Chat Completions API.

    This function prepares the appropriate arguments and delegates the actual API call
    to `call_api_async`, using the global OpenAI client if no client is specified.

    It applies project-standard validation to the response and ensures consistent,
    structured output for downstream use.

    Args:
        prompt: The user/system prompt to send to the LLM.
        model_id: Model identifier (e.g., "gpt-4-turbo", "gpt-4o"). Defaults to GPT_4_1_MINI.
        expected_res_type: Expected output type ("json", "str", "code", etc.). Defaults to "str".
        json_type: The JSON submodel for validation, if applicable (e.g., "tool_selection").
        temperature: Sampling temperature for the LLM. Defaults to 0.4.
        max_tokens: Maximum tokens to generate. Defaults to 1056.
        client: Optional AsyncOpenAI client instance. If not provided, uses the global client.
        response_model (Type[BaseModel], optional): Pydantic model to validate output.

    Returns:
        A validated, structured response model (JSONResponse, TextResponse, CodeResponse,
        ToolCall, or ToolChain), depending on the prompt and output format.

    Raises:
        ValueError: For invalid or failed responses.
        Exception: For unexpected API or network errors.

    Example:
        >>> await call_openai_api_async("Summarize this config.", expected_res_type="json")
    """
    if client is None:
        client = OPENAI_CLIENT

    logger.info("OpenAI client ready for async API call.")

    return await call_api_async(
        client=client,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        # json_type=json_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider=OPENAI,
        response_model=response_model,
    )


# Async wrapper for Anthropic
async def call_anthropic_api_async(
    prompt: str,
    model_id: str = CLAUDE_HAIKU,  # default to haiku (cheapest model)
    expected_res_type: str = "str",
    # json_type: str = "tool_selection",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[AsyncAnthropic] = None,
    response_model: Optional[Type[BaseModel]] = None,
) -> Union[JSONResponse, TextResponse, CodeResponse, ToolCall, ToolChain]:
    """
    Asynchronous convenience wrapper for calling the Anthropic Messages API.

    This function prepares the appropriate arguments and delegates the actual API call
    to `call_api_async`, using the global Anthropic client if no client is specified.

    It applies project-standard validation to the response and ensures consistent,
    structured output for downstream use.

    Args:
        prompt: The user/system prompt to send to the LLM.
        model_id: Model identifier (e.g., "claude-3-haiku", "claude-3-opus").
            Defaults to CLAUDE_HAIKU.
        expected_res_type: Expected output type ("json", "str", "code", etc.). Defaults to "str".
        json_type: The JSON submodel for validation, if applicable (e.g., "tool_selection").
        temperature: Sampling temperature for the LLM. Defaults to 0.4.
        max_tokens: Maximum tokens to generate. Defaults to 1056.
        client: Optional AsyncAnthropic client instance. If not provided, uses the global client.
        response_model (Type[BaseModel], optional): Pydantic model to validate output.

    Returns:
        A validated, structured response model (JSONResponse, TextResponse, CodeResponse,
        ToolCall, or ToolChain), depending on the prompt and output format.

    Raises:
        ValueError: For invalid or failed responses.
        Exception: For unexpected API or network errors.

    Example:
        >>> await call_anthropic_api_async("Extract secrets from this log.", expected_res_type="json")
    """
    if client is None:
        client = ANTHROPIC_CLIENT

    logger.info("Anthropic client ready for async API call.")
    return await call_api_async(
        client=client,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider=ANTHROPIC,
        response_model=response_model,
    )
