"""
File: agent_response_validator.py
Last updated: 2025-07-30

Validation and normalization utilities for agent/LLM responses.

This module standardizes the processing and validation of core agent response types:
- text
- json (arbitrary dict/list)
- code (string/code block)
It also handles tool plan/result extraction and intent classification if required.

Key functions:
    - validate_response_type: Main entry; validates/structures raw agent/LLM output by expected type.
    - clean_and_extract_json: Robustly extracts a JSON object or array from noisy text.
    - validate_tool_plan: Parses and validates tool call plans (list of tool call dicts).
    - validate_intent_response: Validates intent classification output.

All return Pydantic model instances from agent_response_models.py.
"""

import re
import json
import logging
from typing import Any, Dict, List, Union, Optional, Tuple, Type
from agent_models.llm_response_models import *


logger = logging.getLogger(__name__)


def clean_and_extract_json(response_content: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Extracts and parses the first valid JSON object or array from text.

    Handles common cases where LLM output contains extraneous pre/post text, markdown, etc.
    Removes JavaScript-style comments and trailing commas.

    Args:
        response_content (str): Raw LLM or agent response.

    Returns:
        dict or list: Parsed JSON content.

    Raises:
        ValueError: If no valid JSON block can be found.
    """
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        logger.debug("Direct JSON parse failed; attempting fallback extraction.")

    # Try to extract the first object or array
    match = re.search(r"(\{.*?\}|\[.*\])", response_content, re.DOTALL)
    if not match:
        logger.error("No JSON-like content found in response.")
        raise ValueError("No JSON-like content found in response.")

    json_str = match.group(1)
    # Remove trailing commas
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        raise ValueError("Failed to parse JSON from extracted content.")


def is_valid_llm_response(
    obj: Any, model_types: Optional[Tuple[Type, ...]] = None
) -> bool:
    """
    Checks if obj is an instance of any of the allowed LLM response models.

    Args:
        obj (Any): The object to check.
        model_types (tuple[Type], optional): Allowed model types.
            Defaults to standard agent response models.

    Returns:
        bool: True if obj is an allowed model, False otherwise.
    """
    if model_types is None:
        model_types = (
            CodeResponse,
            JSONResponse,
            TextResponse,
            ToolSelect,
            ToolSteps,
        )
    return isinstance(obj, model_types)


def validate_response_type(
    response_content: Any, expected_res_type: str
) -> Union[TextResponse, CodeResponse, JSONResponse]:
    """
    Validates and normalizes the agent/LLM output according to expected type.

    Args:
        response_content: The raw LLM/agent output (string, dict, or list).
        expected_res_type (str): One of "text", "json", "code".

    Returns:
        TextResponse, CodeResponse, or JSONResponse (all Pydantic models).

    Raises:
        ValueError: If parsing or type coercion fails.
    """
    if expected_res_type == "json":
        if isinstance(response_content, (dict, list)):
            return JSONResponse(data=response_content)
        elif isinstance(response_content, str):
            data = clean_and_extract_json(response_content)
            return JSONResponse(data=data)
        else:
            raise ValueError(f"Cannot parse JSON from type: {type(response_content)}")

    elif expected_res_type == "code":
        code = response_content.strip()
        # Remove triple backtick markdown if present
        if code.startswith("```"):
            code = re.sub(r"^```[a-zA-Z]*\n?", "", code)
            if code.endswith("```"):
                code = code[:-3].strip()
        return CodeResponse(code=code)

    elif expected_res_type == "text":
        if not isinstance(response_content, str):
            raise ValueError("Expected string for text response.")
        return TextResponse(content=response_content.strip())

    else:
        raise ValueError(f"Unsupported response type: {expected_res_type}")


def validate_tool_selection_or_steps(
    data: Any,
) -> Union["ToolSelect", "ToolSteps"]:
    """
    Validates and parses a tool-calling response as either a single ToolSelect
        or ToolSteps.

    Args:
        data (Any): Could be a dict (single tool call), a list (multiple calls),
                    a ToolSelect, or ToolSteps.

    Returns:
        ToolSelect or ToolSteps: The validated model instance.

    Raises:
        ValueError: If the input data cannot be parsed as a valid tool
            selection or steps.
    """
    # Already a ToolSelect or ToolSteps? Return as is
    if isinstance(data, (ToolSelect, ToolSteps)):
        return data

    # If data is a dict and looks like a tool call
    if isinstance(data, dict) and "tool" in data and "params" in data:
        try:
            return ToolSelect(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse as ToolSelect: {e}")

    # If data is a list, treat as multi-step
    if isinstance(data, list):
        try:
            steps = [
                ToolSelect(**item) if not isinstance(item, ToolSelect) else item
                for item in data
            ]
            return ToolSteps(steps=steps)
        except Exception as e:
            raise ValueError(f"Failed to parse as ToolSteps: {e}")

    raise ValueError(
        f"Data is not valid for tool selection or steps. Got: {type(data)}, value: {data!r}"
    )


def validate_intent_response(
    response_content: Any,
) -> IntentClassificationResponse:
    """
    Validates and parses an intent classification output.

    Args:
        response_content: Should be a string intent, or dict with "intent" key.

    Returns:
        IntentClassificationResponse

    Raises:
        ValueError: If the intent can't be extracted.
    """
    if isinstance(response_content, str):
        intent = response_content.strip()
    elif isinstance(response_content, dict):
        intent = response_content.get("intent")
    else:
        raise ValueError("Cannot extract intent from response.")

    if not intent or not isinstance(intent, str):
        raise ValueError("Intent must be a non-empty string.")

    return IntentClassificationResponse(intent=intent, status="success")


# --- (Optional) Add more validators for tool results, etc., as agent grows ---
