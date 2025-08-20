"""
tools/llm_response_async.py
Direct LLM response tool with hardened normalization.

Fixes:
- Always unwrap validator/helper objects to plain text.
- If a weird helper (e.g., _LLMResult) slips through, extract attributes
  like `.text`, `.content`, or `.choices[0].message.content` before
  falling back to str().
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from agent.planner import Planner
from agent_models.agent_models import TextResponse
from agent_models.step_status import StepStatus
from tools.tool_models import LLMResponseResult, ToolOutput

logger = logging.getLogger(__name__)


def _coerce_to_text(obj: Any, default: str = "") -> str:
    """Best-effort extraction of text from many SDK/validator shapes."""
    if obj is None:
        return default

    # Pydantic TextResponse
    if isinstance(obj, TextResponse):
        return getattr(obj, "text", None) or getattr(obj, "content", None) or default

    # dict-like payloads
    if isinstance(obj, dict):
        for k in ("text", "content", "message"):
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                return obj[k].strip()
        # OpenAI-like
        try:
            return (
                obj["choices"][0]["message"]["content"]
            ).strip()
        except Exception:
            pass

    # SDK objects with attributes
    for attr in ("text", "content"):
        try:
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            pass

    # OpenAI Chat completion object-ish
    try:
        choices = getattr(obj, "choices", None)
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg:
                c = getattr(msg, "content", None)
                if isinstance(c, str) and c.strip():
                    return c.strip()
    except Exception:
        pass

    return str(obj).strip() or default


def run(prompt: str, planner: Planner) -> Dict[str, Any]:
    """
    Synchronous entry-point required by your ToolRegistry.
    """
    try:
        result = asyncio.run(_generate_text_async(prompt, planner))
        text = _coerce_to_text(result, default=prompt)

        output = ToolOutput(
            status=StepStatus.SUCCESS,
            message="Successfully generated LLM response.",
            result=LLMResponseResult(text=text),
        )
        return output.model_dump()

    except Exception as e:
        logger.exception(f"llm_response_async failed: {e}")
        output = ToolOutput(
            status=StepStatus.ERROR,
            message=f"LLM error: {e}",
            result=None,
        )
        return output.model_dump()


async def _generate_text_async(prompt: str, planner: Planner) -> TextResponse:
    """
    Use the planner's universal dispatcher which now properly sends system+user.
    """
    return await planner.dispatch_llm_async(
        user_prompt=prompt,
        system_prompt="You are a helpful assistant.",
        response_type="text",
        response_model=TextResponse,
    )

