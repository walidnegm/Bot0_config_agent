"""
agent/intent_classifier.py

Intent classification utilities for agent task routing and decomposition.

This module provides async classifier functions for agent pipelines:
- Classifies whether a user instruction is a project description request or a more complex task.
- Distinguishes between single-step and multi-step tasks to drive planner logic.

Example Usage:
    import asyncio
    from agent.planner import Planner

    import logging
    logging.basicConfig(level=logging.INFO)

    # Instantiate your planner with the appropriate model
    planner = Planner(local_model_name="llama_2_7b_chat")

    # Classify project description intent
    print("Describe only:")
    print(asyncio.run(classify_describe_only("Describe this project.", planner)))

    # Classify task decomposition (single-step vs. multi-step)
    print("Task decomposition:")
    print(
        asyncio.run(
            classify_task_decomposition_async(
                "Summarize every Python file in this 200-file repo.", planner
            )
        )
    )
"""

from pathlib import Path
from typing import Optional
import logging
import yaml
from prompts.load_agent_prompts import (
    load_describe_only_prompt,
    load_task_decomposition_prompt,
)
from agent.planner import Planner  # <-- import your Planner with dispatch_llm_async


logger = logging.getLogger(__name__)


async def classify_describe_only_async(
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
        cfg = load_describe_only_prompt(instruction)
        full_prompt = (
            cfg["system_prompt"].strip()
            + "\n"
            + cfg["describe_only_prompt"].strip()
            + "\n"
            + cfg["user_prompt_template"].strip()
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


async def classify_task_decomposition_async(
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
        cfg = load_task_decomposition_prompt(instruction)
        full_prompt = (
            cfg["system_prompt"].strip()
            + "\n"
            + cfg["single_vs_multi_step_prompt"].strip()
            + "\n"
            + cfg["user_prompt_template"].strip()
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
