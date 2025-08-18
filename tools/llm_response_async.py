import logging
import asyncio
from typing import Dict

# Import the Planner and helper models
from agent.planner import Planner
from agent_models.agent_models import TextResponse
from agent_models.step_status import StepStatus
from tools.tool_models import LLMResponseResult, ToolOutput

logger = logging.getLogger(__name__)

def llm_response_async(*, prompt: str, local_model_name: str = None, api_model_name: str = None, **kwargs) -> Dict:
    """
    Synchronous tool wrapper that intelligently routes a prompt to either a
    local LLM or a cloud API based on which model name is provided.
    """
    try:
        # ####################################################################
        # UPDATED LOGIC: Instantiate a Planner to handle the dispatch.
        # This allows the tool to correctly use either a local or API model.
        # ####################################################################
        planner = Planner(local_model_name=local_model_name, api_model_name=api_model_name)

        # Run the async helper function in a new event loop.
        result = asyncio.run(_generate_text_async(prompt, planner))

        if isinstance(result, TextResponse):
            text = getattr(result, "text", "") or getattr(result, "content", "") or prompt
        else:
            text = str(result)

        # Use Pydantic models for a clean, validated return structure.
        output = ToolOutput(
            status=StepStatus.SUCCESS,
            message="Successfully generated LLM response.",
            result=LLMResponseResult(text=text)
        )
        return output.model_dump()

    except Exception as e:
        logger.exception(f"llm_response_async failed: {e}")
        output = ToolOutput(
            status=StepStatus.ERROR,
            message=f"LLM error: {e}",
            result=None
        )
        return output.model_dump()

async def _generate_text_async(prompt: str, planner: Planner) -> TextResponse:
    """
    Async helper that uses the planner's universal dispatch method to generate text.
    """
    try:
        # Use the planner's dispatch method, which can handle any model.
        response = await planner.dispatch_llm_async(
            user_prompt=prompt,
            system_prompt="You are a helpful assistant.",
            response_type="text",
            response_model=TextResponse,
        )
        return response
    except Exception as e:
        logger.error(f"LLM generation failed during async dispatch: {e}")
        raise
