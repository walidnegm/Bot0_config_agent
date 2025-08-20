"""
agent/intent_classifier_core.py
Core intent classification utilities for agent task routing.
"""

import re
import json
import logging
from agent.llm_manager import LLMManager
from utils.get_llm_api_keys import get_openai_api_key

logger = logging.getLogger(__name__)

def classify_describe_only(instruction: str, use_openai: bool = False) -> str:
    """
    Classifies the intent of the instruction as either 'describe_project' or 'unknown'.
    Uses OpenAI if use_openai=True, otherwise uses the local model via LLMManager.

    OpenAI must return exactly: describe_project OR unknown (as plain text).
    """
    if use_openai:
        try:
            get_openai_api_key()  # Validate API key presence
        except ValueError as e:
            logger.error(f"[IntentClassifier] Missing OpenAI API key: {e}")
            return "error"
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
        from utils.llm_api_async import call_openai_api_async
        try:
            raw = asyncio.run(call_openai_api_async(
                model_id="gpt-4.1-mini",
                prompt=prompt,
                response_format="text"
            ))
        except Exception as e:
            logger.error(f"[IntentClassifier] OpenAI call failed: {e}")
            return "error"
    else:
        prompt = (
            "You are a strict classifier. If the instruction is asking to summarize, describe, "
            "or give an overview of the project, return: describe_project.\n"
            "If not, return: unknown.\n"
            f"Instruction: {instruction}\nIntent:"
        )
        try:
            raw = LLMManager().generate(prompt)
        except Exception as e:
            logger.error(f"[IntentClassifier] Local LLM call failed: {e}")
            return "error"

    try:
        result = raw.get("text") if isinstance(raw, dict) else raw
        result = result.strip()

        logger.debug(f"[IntentClassifier] Raw model output: {repr(result)}")

        # For OpenAI: reject anything that looks like JSON or tools
        if use_openai:
            if result.startswith("[") or result.startswith("{"):
                logger.warning("[IntentClassifier] Rejected invalid structured output from OpenAI.")
                return "unknown"
            result = result.lower().strip()
            return result if result in {"describe_project", "unknown"} else "unknown"

        # For local LLM: extract from assistant block or clean fallback
        match = re.search(
            r"<\|im_start\|>assistant\s+(.*?)\s*<\|im_end\|>", result, re.DOTALL
        )
        if match:
            result = match.group(1).strip().lower()
        else:
            result = result.replace("<|im_end|>", "").strip().lower()

        logger.info(f"[IntentClassifier] Parsed intent: {repr(result)}")

        return result if result in {"describe_project", "unknown"} else "unknown"

    except Exception as e:
        logger.error(f"[IntentClassifier] Error during classification: {e}")
        return "error"
