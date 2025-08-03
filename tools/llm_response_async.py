import logging
from typing import Dict
import itertools
from agent_models.llm_response_validators import is_valid_llm_response

logger = logging.getLogger(__name__)


async def llm_response_async(
    *, planner, user_prompt, system_prompt="", temperature=0.3, **kwargs
) -> Dict:  #! Enforcing keyword-only argument b/c this tool needs CLARITY!
    """
    Tool: LLM Response (agent-native, async)

    Args:
        planner (Planner): The Planner instance to use for LLM dispatch.
        user_prompt (str): The main user/task prompt.
        system_prompt (str, optional): System/context prompt.
        temperature (float, optional): Temperature for generation.

    Returns:
        dict: { "status": "ok"|"error", "message": response text or error,
            "result"|"data"|"content"...: response }
    """
    try:
        # Call the planner's dispatch_llm_async method (fully agent-native)
        response = await planner.dispatch_llm_async(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_type=None,  # Let the planner infer
            response_model=None,  # Let the planner infer
            temperature=temperature,
        )

        if not is_valid_llm_response(response):
            raise ValueError(f"Invalid LLM response type: {type(response)}")

        response_dict = response.model_dump()

        logger.debug(f"LLM response pyd model type: {type(response).__name__}")
        logger.debug(
            f"Preview: response data after model dump\n{dict(itertools.islice(response_dict.items(), 3))}"
        )  # Shows a dict with first 3 items

        return response_dict

    except Exception as e:
        return {"status": "error", "message": f"LLM error: {e}"}
