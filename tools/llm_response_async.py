# tools/llm_response_async.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

def run(prompt: str, planner: Any) -> Optional[str]:
    try:
        return asyncio.run(_generate_text_async(prompt, planner))
    except Exception as e:
        logger.exception("llm_response_async failed: %s", e)
        return None

async def _generate_text_async(prompt: str, planner: Any) -> str:
    system = "You are a helpful assistant. Answer succinctly."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    generation_kwargs = {}

    resp = await planner.dispatch_llm_async(
        messages=messages,
        temperature=0.3,
        generation_kwargs=generation_kwargs,
    )

    text = (resp or {}).get("text", "")
    return text
