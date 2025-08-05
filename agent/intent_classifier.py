# agent/intent_classifier.py
# agent/intent_classifier.py
from pathlib import Path
import re
import yaml
from typing import Optional
import logging

from agent.planner import Planner  # <-- import your Planner with dispatch_llm_async
from configs.paths import AGENT_PROMPTS

logger = logging.getLogger(__name__)


def load_intent_templates(path: Path = AGENT_PROMPTS):
    """Load intent classifier prompt templates from YAML file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)["intent_classifier"]
    except Exception as e:
        logger.error(f"[IntentClassifier] Failed to load YAML: {e}")
        raise


# Load once at module level
try:
    prompts = load_intent_templates()
except Exception as e:
    prompts = {}
    logger.error(
        "[IntentClassifier] Failed to load intent classifier prompts at import."
    )


async def classify_describe_only(
    instruction: str,
    planner: "Planner",
) -> str:
    """
    Classify if the instruction is asking for a project description or not.

    Args:
        instruction (str): The user instruction.
        planner (Planner): An instance of Planner with dispatch_llm_async.

    Returns:
        str: "describe_project", "unknown", or "error".
    """
    try:
        cfg = prompts["describe_only"]
        full_prompt = (
            cfg["system_prompt"].strip()
            + "\n"
            + cfg["describe_only_prompt"].strip()
            + "\n"
            + cfg["user_prompt_template"].format(user_task=instruction).strip()
        )
        # Use planner's dispatch_llm_async for all LLM calls
        result_obj = await planner.dispatch_llm_async(
            user_prompt=full_prompt,
            system_prompt="",  # Not needed since in template; or pass cfg["system_prompt"]
            response_type="text",
        )

        # Defensive: prefer .text, fallback to str
        result = getattr(result_obj, "text", None) or str(result_obj)
        result = result.strip().lower()
        # Remove label if present (defensive)
        if result.startswith("instruction:"):
            result = result.replace("instruction:", "").strip()
        if result in {"describe_project", "unknown"}:
            logger.info(f"[IntentClassifier] describe_only result: {result}")
            return result
        logger.warning(f"[IntentClassifier] Unexpected output: {result!r}")
        return "unknown"
    except Exception as e:
        logger.error(f"[IntentClassifier] classify_describe_only failed: {e}")
        return "error"


async def classify_task_decomposition(
    instruction: str,
    planner: "Planner",
) -> str:
    """
    Classify if the instruction should be handled in a single step or needs multi-step planning.

    Args:
        instruction (str): The user instruction.
        planner (Planner): An instance of Planner with dispatch_llm_async.

    Returns:
        str: "single-step", "multi-step", or "unknown"/"error".
    """
    try:
        cfg = prompts["task_decomposition"]
        full_prompt = (
            cfg["system_prompt"].strip()
            + "\n"
            + cfg["single_vs_multi_step_prompt"].strip()
            + "\n"
            + cfg["user_prompt_template"].format(user_task=instruction).strip()
        )

        result_obj = await planner.dispatch_llm_async(
            user_prompt=full_prompt,
            system_prompt="",  # Not needed; included in template
            response_type="text",
        )

        result = getattr(result_obj, "text", None) or str(result_obj)
        result = result.strip().lower()
        # Defensive clean-up
        if "single-step" in result:
            logger.info(f"[IntentClassifier] task_decomposition: single-step")
            return "single-step"
        if "multi-step" in result:
            logger.info(f"[IntentClassifier] task_decomposition: multi-step")
            return "multi-step"
        logger.warning(f"[IntentClassifier] Unexpected output: {result!r}")
        return "unknown"
    except Exception as e:
        logger.error(f"[IntentClassifier] classify_task_decomposition failed: {e}")
        return "error"


# Example usage (async)
if __name__ == "__main__":
    import asyncio
    from agent.planner import Planner

    logging.basicConfig(level=logging.INFO)
    # You'd pass the correct local_model_name/api_model_name here as used in your CLI
    planner = Planner(local_model_name="llama_2_7b_chat")
    print("Describe only:")
    print(asyncio.run(classify_describe_only("Describe this project.", planner)))
    print("Task decomposition:")
    print(
        asyncio.run(
            classify_task_decomposition(
                "Summarize every Python file in this 200-file repo.", planner
            )
        )
    )
