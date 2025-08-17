import logging
from typing import Dict
from agent_models.step_status import StepStatus

logger = logging.getLogger(__name__)

def llm_response_async(*, prompt: str, **kwargs) -> Dict:
    """
    Synchronous tool wrapper expected by ToolRegistry/Executor.

    NOTE: The current executor invokes tools synchronously (no await),
    so this function MUST be sync and return a plain dict ToolOutput
    shape: {"status", "message", "result": {...}}.
    """
    try:
        # Minimal placeholder: echo back the prompt as "generated" text.
        # You can wire a real LLM call here later (or update the executor to await coroutines).
        text = _generate_text(prompt)
        return {
            "status": StepStatus.SUCCESS,
            "message": "",
            "result": {"text": text},
        }
    except Exception as e:
        logger.exception("llm_response_async failed")
        return {
            "status": StepStatus.ERROR,
            "message": f"LLM error: {e}",
            "result": None,
        }

def _generate_text(prompt: str) -> str:
    # Simple, safe default. Replace with a real model call if desired.
    return (
        "Python is a high-level, general-purpose programming language focused on readability "
        "and a rich standard library. It’s widely used for web backends, data science, "
        "automation, scripting, and more."
        if prompt.strip().lower() == "what is python?"
        else prompt
    )

