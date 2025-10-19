# tools/llm_response_async.py
"""
Async LLM text generation using OpenAI.
Now MCP-compatible and safe inside existing asyncio event loops.
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import Dict, Any
from agent_models.step_status import StepStatus
from dotenv import load_dotenv
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
load_dotenv()  # ✅ loads .env if present


# ---------------------------------------------------------------------
# Core async generator
# ---------------------------------------------------------------------
async def _generate_response(prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("[llm_response_async] Missing OPENAI_API_KEY in environment.")
        return {
            "status": StepStatus.ERROR,
            "message": "Missing OPENAI_API_KEY in environment.",
            "result": None,
        }

    try:
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise and helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        return {
            "status": StepStatus.SUCCESS,
            "message": "Response generated successfully.",
            "result": text,
        }
    except Exception as e:
        logger.exception("[llm_response_async] API call failed: %s", e)
        return {"status": StepStatus.ERROR, "message": str(e), "result": None}


# ---------------------------------------------------------------------
# MCP entrypoints
# ---------------------------------------------------------------------
def get_tool_definition():
    return {
        "name": "llm_response_async",
        "description": "Generates a text completion using OpenAI asynchronously.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The text prompt to complete."}
            },
            "required": ["prompt"],
        },
    }


def run(params: Dict[str, Any] | None = None):
    """MCP entrypoint wrapper that works both inside and outside event loops."""
    params = params or {}
    prompt = params.get("prompt", "")

    try:
        loop = asyncio.get_running_loop()
        # ✅ We're already inside an event loop (MCP runtime)
        return asyncio.ensure_future(_generate_response(prompt))
    except RuntimeError:
        # ✅ No running loop → safe to run normally
        return asyncio.run(_generate_response(prompt))

