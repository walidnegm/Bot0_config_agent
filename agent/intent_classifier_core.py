# agent/intent_classifier_core.py

from agent.llm_manager import LLMManager
from agent import llm_openai

# Use OpenAI or local model based on context
llm = None


def classify_describe_only(instruction: str, use_openai: bool = False) -> str:
    global llm
    if llm is None:
        if use_openai:
            llm = llm_openai
        else:
            llm = LLMManager()

    prompt = (
        "You are a strict classifier. If the instruction is asking to summarize, describe, or give an overview of the project, return: describe_project.\n"
        "If not, return: unknown.\n"
        f"Instruction: {instruction}\n"
        "Intent:"
    )

    try:
        result = llm.generate(prompt).strip().lower()
        if "describe_project" in result:
            return "describe_project"
        return "unknown"
    except Exception as e:
        return "error"

