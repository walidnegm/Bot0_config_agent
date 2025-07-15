# agent/intent_classifier_core.py

from agent.llm_manager import LLMManager
from agent import llm_openai


def classify_describe_only(instruction: str, use_openai: bool = False) -> str:
    """
    Classifies the intent of the instruction as either 'describe_project' or 'unknown',
    using OpenAI if `use_openai` is True, otherwise falling back to the local LLMManager.
    """
    prompt = (
        "You are a strict classifier. If the instruction is asking to summarize, describe, "
        "or give an overview of the project, return: describe_project.\n"
        "If not, return: unknown.\n"
        f"Instruction: {instruction}\n"
        "Intent:"
    )

    try:
        if use_openai:
            result = llm_openai.generate(prompt).strip().lower()
        else:
            result = LLMManager().generate(prompt).strip().lower()

        if "describe_project" in result:
            return "describe_project"
        return "unknown"
    except Exception:
        return "error"

