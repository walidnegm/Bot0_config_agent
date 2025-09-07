"""
bot0_config_agent/tools/tool_scripts/llm_response_async.py

Agent-native LLM call helper (sync wrapper around an async planner method).
"""

import asyncio
import itertools
import logging
from typing import Dict, Any, Optional

from bot0_config_agent.utils.llm.llm_response_validators import is_valid_llm_response
from bot0_config_agent.agent_models.step_status import StepStatus

logger = logging.getLogger(__name__)


def _extract_text_from_response(resp_model: Any) -> str:
    """
    Best-effort extraction of plain text from a validated LLM response model.
    Falls back to str(model_dump()) if no obvious text field exists.
    """
    try:
        # Pydantic model → dict
        d = resp_model.model_dump()
        # common fields to try in order
        for key in ("text", "result", "content", "message", "output"):
            v = d.get(key)
            if isinstance(v, str) and v.strip():
                return v
        # sometimes nested under data.{text|content}
        data = d.get("data")
        if isinstance(data, dict):
            for key in ("text", "content", "message"):
                v = data.get(key)
                if isinstance(v, str) and v.strip():
                    return v
        # nothing obvious → pretty-print first few items
        head = dict(itertools.islice(d.items(), 5))
        return str(head)
    except Exception:
        # last resort
        return str(resp_model)


def llm_response_async(
    **kwargs,
) -> Dict[str, Any]:  #! Enforcing keyword-only argument b/c this tool needs CLARITY!
    """
    Call the project's LLM via the Planner's async dispatch, but expose a
    synchronous tool API so the executor can call it normally.

    Args (kwargs):
        prompt (str): User/task prompt (required).
        system_prompt (str, optional): System/context prompt. Default "".
        temperature (float, optional): Sampling temperature. Default 0.3.
        planner (Planner, optional): If provided and has `dispatch_llm_async`,
            this tool will call it via `asyncio.run(...)`.

    Returns:
        dict: Standard tool envelope (status, message, result[str]).
    """
    prompt: Optional[str] = kwargs.get("prompt")
    system_prompt: str = kwargs.get("system_prompt", "")
    temperature: float = kwargs.get("temperature", 0.3)
    planner = kwargs.get("planner")

    if not prompt or not isinstance(prompt, str):
        return {
            "status": StepStatus.ERROR,
            "message": "Missing required 'prompt' (str).",
            "result": None,
        }

    # If no planner provided, be explicit (this tool is agent-native).
    if planner is None or not hasattr(planner, "dispatch_llm_async"):
        return {
            "status": StepStatus.ERROR,
            "message": "Planner with 'dispatch_llm_async' is required for llm_response_async.",
            "result": None,
        }

    try:
        # Run the async planner method in a local event loop.
        async def _run():
            return await planner.dispatch_llm_async(
                user_prompt=prompt,
                system_prompt=system_prompt,
                response_type=None,
                response_model=None,
                temperature=temperature,
            )

        response_model = asyncio.run(_run())

        if not is_valid_llm_response(response_model):
            raise ValueError(f"Invalid LLM response type: {type(response_model)}")

        # Logging preview (non-intrusive)
        rd = response_model.model_dump()
        logger.debug(
            f"[llm_response_async] LLM response model type: {type(response_model).__name__}"
        )
        logger.debug(
            f"[llm_response_async] Preview: {dict(itertools.islice(rd.items(), 3))}"
        )

        text = _extract_text_from_response(response_model)
        return {
            "status": StepStatus.SUCCESS,
            "message": "LLM responded.",
            "result": Any,
        }

    except Exception as e:
        return {
            "status": StepStatus.ERROR,
            "message": f"LLM error: {e}",
            "result": None,
        }
