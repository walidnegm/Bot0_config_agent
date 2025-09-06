"""
File: utils/llm/llm_response_validator.py
Last updated: 2025-07-30

Validation and normalization utilities for agent/LLM responses.

This module standardizes the processing and validation of core agent
response types:
- text
- json (arbitrary dict/list)
- code (string/code block)
It also handles tool plan/result extraction and intent classification
if required.

Key functions:
    - validate_response_type: Main entry; validates/structures raw agent/LLM
        output by expected type.
    - clean_and_extract_json: Robustly extracts a JSON object or array
        from noisy text.
    - validate_tool_plan: Parses and validates tool call plans
        (list of tool call dicts).
    - validate_intent_response: Validates intent classification output.

All return Pydantic model instances from agent_response_models.py.
"""

import re
import json
import logging
from typing import Any, Dict, Iterable, List, Union, Optional, Sequence, Tuple, Type
from bot0_config_agent.agent_models.agent_models import *


logger = logging.getLogger(__name__)

# Match ANY <tag>...</tag> where the closing tag matches the opener.
_TAG_BLOCK_RE = re.compile(
    r"<(?P<tag>[A-Za-z0-9_\-]+)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    re.DOTALL,
)


def _normalize_allowed_tags(
    allowed: Optional[Union[str, Sequence[str]]],
) -> Optional[set]:
    if allowed is None:
        return None
    if isinstance(allowed, str):
        return {allowed.strip().lower()}
    return {str(t).strip().lower() for t in allowed}


def extract_last_xml_tag_block(
    text: str,
    allowed_tags: Optional[Union[str, Sequence[str]]] = (
        "result",
        "answer",
        "code",
        "output",
        "final",
        "final_output",
    ),
) -> Optional[str]:
    """
    Returns the inner text of the LAST fully-closed <tag>...</tag> block
    if the tag is in allowed_tags. Otherwise None.
    """
    if not text:
        return None

    allowed = _normalize_allowed_tags(allowed_tags)
    last = None
    for m in _TAG_BLOCK_RE.finditer(text):
        tag = m.group("tag").strip().lower()
        if (allowed is None) or (tag in allowed):
            last = m.group("body")

    return last.strip() if last is not None else None


def _strip_backticks(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove opening fence with optional language token
        s = re.sub(r"^```[A-Za-z0-9_\-]*\n?", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _strip_last_tag_then_backticks(
    s: str, allowed_tags: Optional[Union[str, Sequence[str]]] = None
) -> str:
    """Helper: strip last allowed <tag>...</tag> if present, then remove code fences."""
    inner = extract_last_xml_tag_block(s, allowed_tags=allowed_tags)
    s2 = inner if inner is not None else s
    return _strip_backticks(s2)


def _clean_trailing_commas(json_str: str) -> str:
    # cautious trailing-comma cleaner (object/array ends)
    return re.sub(r",\s*([\]}])", r"\1", json_str)


def clean_and_extract_json(response_content: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Extracts and parses the first valid JSON object or array from text.

    Handles common cases where LLM output contains extraneous pre/post text,
    markdown, etc. Normalizes trailing commas. (Comment removal optional; see below.)

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
    matches = list(re.finditer(r"(\{.*?\}|\[.*?\])", response_content, re.DOTALL))
    if not matches:
        logger.error("No JSON-like content found in response.")
        raise ValueError("No JSON-like content found in response.")

    json_str = matches[-1].group(1)

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
            ToolCall,
            ToolChain,
        )
    return isinstance(obj, model_types)


def validate_response_type(
    response_content: Any,
    expected_res_type: str,
    xml_tag: Optional[Union[str, Sequence[str]]] = None,
) -> Union[TextResponse, CodeResponse, JSONResponse]:
    """
    Validates and normalizes the agent/LLM output according to expected type.

    Args:
        response_content: The raw LLM/agent output (string, dict, or list).
        expected_res_type (str): One of "text", "json", "code".
        xml_tag (Optional[Union[str, Sequence[str]]]):
            xml tag deliminator to parse output/results/...

    Returns:
        TextResponse, CodeResponse, or JSONResponse (all Pydantic models).

    Raises:
        ValueError: If parsing or type coercion fails.
    """
    if expected_res_type == "json":
        if isinstance(response_content, (dict, list)):
            return JSONResponse(data=response_content)
        if isinstance(response_content, str):
            data = clean_and_extract_json(response_content)
            return JSONResponse(data=data)
        raise ValueError(f"Cannot parse JSON from type: {type(response_content)}")

    if expected_res_type == "code":
        if not isinstance(response_content, str):
            raise ValueError("Expected string for code response.")

        if xml_tag:
            code = _strip_last_tag_then_backticks(
                response_content, allowed_tags=xml_tag
            )
        else:
            code = _strip_backticks(response_content)

        return CodeResponse(code=code)

    if expected_res_type == "text":
        if not isinstance(response_content, str):
            raise ValueError("Expected string for text response.")

        if xml_tag:
            # only strip if caller explicitly specifies tags
            text = _strip_last_tag_then_backticks(
                response_content, allowed_tags=xml_tag
            )
        else:
            # just strip backticks / fences, keep content as-is
            text = _strip_backticks(response_content)

        return TextResponse(content=text)

    raise ValueError(f"Unsupported response type: {expected_res_type}")


def validate_tool_selection_or_steps(
    data: Any,
) -> Union["ToolCall", "ToolChain"]:
    """
    Validates and parses a tool-calling response as either a single ToolCall
        or ToolChain.

    Args:
        data (Any): Could be a dict (single tool call), a list (multiple calls),
                    a ToolCall, or ToolChain.

    Returns:
        ToolCall or ToolCahin: The validated model instance.

    Raises:
        ValueError: If the input data cannot be parsed as a valid tool
            selection or steps.
    """
    # Already a ToolCall or ToolChain Return as is
    if isinstance(data, (ToolCall, ToolChain)):
        return data

    # If data is a dict and looks like a tool call
    if isinstance(data, dict) and "tool" in data and "params" in data:
        try:
            return ToolCall(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse as ToolCall: {e}")

    # If data is a list, treat as multi-step
    if isinstance(data, list):
        try:
            steps = [
                ToolCall(**item) if not isinstance(item, ToolCall) else item
                for item in data
            ]
            return ToolChain(steps=steps)
        except Exception as e:
            raise ValueError(f"Failed to parse as ToolChain: {e}")

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
