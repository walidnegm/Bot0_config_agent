"""prompts/load_agent_prompts.py

Robust prompt section loader for multi-agent tool system.
Supports planner, summarizer, evaluator, and intent_classifier sections.
Auto-injects tools into planner. Jinja2 + YAML.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Mapping
from jinja2 import Environment, FileSystemLoader
import yaml
from configs.paths import AGENT_PROMPTS

logger = logging.getLogger(__name__)


def _load_section(
    section: str,
    template_path: Path | str = AGENT_PROMPTS,
    **kwargs,
) -> Dict[str, str]:
    """
    Loads and renders a section of the Jinja2+YAML prompt template.

    Args:
        section (str): Section name to extract from YAML (e.g. "planner", "summarizer").
        template_path (Path|str): Path to the Jinja2 YAML prompt template.
        **kwargs: Variables to inject (user_task, tools, etc.).

    Returns:
        Dict[str, str]: All key/value pairs under the given section.
    """
    template_dir = str(Path(template_path).parent)
    template_name = Path(template_path).name

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    rendered = template.render(**kwargs)
    config = yaml.safe_load(rendered)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in template file.")
    return config[section]


def load_planner_prompts(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, str]:
    """
    Loads the planner prompt section, auto-injecting tool registry.


    Load and render the planner prompt configuration using Jinja2 and YAML.

    Args:
        template_path (Path|str): Path to the Jinja2 YAML prompt template.
        user_task (str): The task the user is asking for; injected into
            the prompt.
        tools (List[Dict[str, Any]]): A list of tool definitions to inject.

    Returns:
        Dict[str, str]: A dictionary containing:
            - system_prompt
            - select_single_tool_prompt
            - select_multi_tool_prompt
            ...
    """
    from tools.tool_registry import (
        ToolRegistry,
    )  # imported here to avoid circular import

    tool_registry = ToolRegistry()
    tools_for_prompt = [
        {
            "name": name,
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {}).get("properties", {}),
            "usage_hint": spec.get("usage_hint", ""),
        }
        for name, spec in tool_registry.get_all().items()
    ]
    return _load_section(
        section="planner",
        template_path=template_path,
        user_task=user_task,
        tools=tools_for_prompt,
    )


def load_summarizer_prompt(
    log_text: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, str]:
    """
    Loads the summarizer prompt configuration from a YAML file.

    Returns:
        Dict[str, str]: Dictionary with keys:
            - system_prompt
            - user_prompt_template
    """
    return _load_section(
        section="summarizer",
        template_path=template_path,
        log_text=log_text,
    )


def load_evaluator_prompt(
    task: str = "",
    response: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, str]:
    """
    Loads the evaluator prompt section.
    """
    return _load_section(
        section="evaluator",
        template_path=template_path,
        task=task,
        response=response,
    )


def load_intent_classifier_prompts(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, str]:
    """
    Loads the intent_classifier prompt section (for both task_decomposition
    and describe_only).
    """
    return _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
    )


def load_task_decomposition_prompt(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Mapping[str, Any]:
    """
    Loads just the task_decomposition subkey from intent_classifier.
    """
    prompts = _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
    )
    return prompts["task_decomposition"]


def load_describe_only_prompt(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Mapping[str, Any]:
    """
    Loads just the describe_only subkey from intent_classifier.
    """
    prompts = _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
    )
    return prompts["describe_only"]


# Static (global) prompt dict for classifier-only (no runtime injection needed)
try:
    PROMPTS = _load_section("intent_classifier")
except Exception as e:
    PROMPTS = {}
    logger.error(
        "[load_agent_prompts] Failed to load global intent_classifier prompts: %s", e
    )
