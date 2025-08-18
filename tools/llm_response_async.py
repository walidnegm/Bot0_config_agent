import logging
from typing import Dict
import asyncio
from agent_models.step_status import StepStatus
from tools.tool_models import LLMResponseResult
from agent.llm_manager import get_llm_manager
from agent_models.agent_models import TextResponse

logger = logging.getLogger(__name__)

def llm_response_async(*, prompt: str, local_model_name: str = None, api_model_name: str = None, **kwargs) -> Dict:
    """
    Synchronous tool wrapper expected by ToolRegistry/Executor.
    Integrates with the LLM manager to generate a response for the given prompt.
    Returns a dict in ToolOutput shape: {"status", "message", "result": {...}}.
    """
    try:
        # Ensure at least one model is specified (local or API)
        if not (local_model_name or api_model_name):
            raise ValueError("Either local_model_name or api_model_name must be provided.")
        # Use local_model_name if provided, else api_model_name
        model_name = local_model_name if local_model_name else api_model_name
        # Initialize LLM manager with the specified model
        llm_manager = get_llm_manager(model_name)
        # Since the executor expects a sync function, run the async LLM call in the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in an async context, use a nested loop or executor
            result = loop.run_until_complete(_generate_text_async(prompt, llm_manager))
        else:
            # Safe to run directly
            result = asyncio.run(_generate_text_async(prompt, llm_manager))
        # Extract text from the response (assuming TextResponse from planner.py)
        if isinstance(result, TextResponse):
            text = getattr(result, "text", "") or getattr(result, "content", "") or prompt
        else:
            text = str(result)  # Fallback for unexpected response types
        return {
            "status": StepStatus.SUCCESS,
            "message": "",
            "result": LLMResponseResult(text=text).model_dump(),
        }
    except Exception as e:
        logger.exception(f"llm_response_async failed: {e}")
        return {
            "status": StepStatus.ERROR,
            "message": f"LLM error: {e}",
            "result": None,
        }

async def _generate_text_async(prompt: str, llm_manager) -> TextResponse:
    """
    Async helper to generate text using the LLM manager.
    """
    try:
        # Call the LLM manager's generate method (sync or async internally handled)
        response = await llm_manager.generate_async(
            user_prompt=prompt,
            system_prompt="",  # Optional: Add a default system prompt if needed
            expected_res_type="text",
            response_model=TextResponse,
        )
        return response
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise
