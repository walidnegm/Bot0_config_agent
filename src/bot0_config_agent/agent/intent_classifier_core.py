import re
import json
from bot0_config_agent.agent.llm_manager import LLMManager
from bot0_config_agent.agent import llm_openai


def classify_describe_only(instruction: str, use_openai: bool = False) -> str:
    """
    Classifies the intent of the instruction as either 'describe_project' or 'unknown'.
    Uses OpenAI if use_openai=True, otherwise uses the local model via LLMManager.

    OpenAI must return exactly: describe_project OR unknown (as plain text).
    """
    if use_openai:
        prompt = (
            "You are a strict intent classifier.\n"
            "Respond with ONLY ONE WORD: either 'describe_project' or 'unknown'.\n"
            "Do NOT return JSON.\n"
            "Do NOT return a list.\n"
            "Do NOT call any tools.\n"
            "Just respond with one word. Examples:\n"
            "- describe this project → describe_project\n"
            "- what is cuda → unknown\n\n"
            f"Instruction: {instruction}\n"
            "Intent:"
        )

    else:
        prompt = (
            "You are a strict classifier. If the instruction is asking to summarize, describe, "
            "or give an overview of the project, return: describe_project.\n"
            "If not, return: unknown.\n"
            f"Instruction: {instruction}\nIntent:"
        )

    try:
        raw = (
            llm_openai.generate(prompt) if use_openai else LLMManager().generate(prompt)
        )

        result = raw.get("text") if isinstance(raw, dict) else raw
        result = result.strip()

        print(f"[IntentClassifier] 🔍 Raw model output:\n{repr(result)}")

        # ✅ For OpenAI: reject anything that looks like JSON or tools
        if use_openai:
            if result.startswith("[") or result.startswith("{"):
                print(
                    "[IntentClassifier] ❌ Rejected invalid structured output from OpenAI."
                )
                return "unknown"
            result = result.lower().strip()
            return result if result in {"describe_project", "unknown"} else "unknown"

        # ✅ For local LLM: extract from assistant block or clean fallback
        match = re.search(
            r"<\|im_start\|>assistant\s+(.*?)\s*<\|im_end\|>", result, re.DOTALL
        )
        if match:
            result = match.group(1).strip().lower()
        else:
            result = result.replace("<|im_end|>", "").strip().lower()

        print(f"[IntentClassifier] ✅ Parsed intent: {repr(result)}")

        return result if result in {"describe_project", "unknown"} else "unknown"

    except Exception as e:
        print(f"[IntentClassifier] ❌ Error during classification: {e}")
        return "error"
