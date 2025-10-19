# tools/llm_response_async.py
"""
Simple helper tool that lets the agent call the LLM directly with a prompt and
return the model's text. This uses Planner.dispatch_llm_async(messages=...)
to stay backend-agnostic (local/API).

Fixes:
- Removed the old 'user_prompt=' call-site. We now pass ChatML-style messages.
- Returns the plain text from the LLMManager {"text": "..."} contract.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Tool signature expected by your ToolRegistry:
# def run(*, prompt: str, planner) -> str

def run(*, prompt: str, planner) -> Optional[str]:
    """
    Execute a direct LLM call with the provided prompt. Returns text.
    """
    try:
        return asyncio.run(_generate_text_async(prompt, planner))
    except Exception as e:
        logger.exception("llm_response_async failed: %s", e)
        # Your executor will capture this as tool error
        return None


async def _generate_text_async(prompt: str, planner) -> str:
    """
    Build messages and call Planner.dispatch_llm_async(messages=...).
    """
    # minimal, stable system message — keeps this tool generic
    system = "You are a helpful assistant. Answer succinctly."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    # You can tweak temperature or pass extra kwargs if desired
    generation_kwargs: Dict[str, Any] = {}

    resp = await planner.dispatch_llm_async(
        messages=messages,
        temperature=0.3,
        generation_kwargs=generation_kwargs,
    )

    # Planner.dispatch_llm_async returns {"text": "..."} from LLMManager
    text = (resp or {}).get("text", "")
    return text

