"""prompts/load_agent_prompts.py

Robust prompt section loader for multi-agent tool system.
Supports planner, summarizer, evaluator, and intent_classifier sections.
Auto-injects tools into planner. Jinja2 + YAML.
"""

import logging
from pathlib import Path
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader
import yaml
from configs.paths import AGENT_PROMPTS
from tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def _to_dict(obj):
    # Pydantic model → dict
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj if isinstance(obj, dict) else {}


def _extract_parameters(spec_dict):
    """
    Return a simple {param: {type: "<type>"}} shape that your Jinja template expects.
    Handles either JSON Schema style {"properties": {...}} or already-flat dicts.
    """
    params = spec_dict.get("parameters") or {}
    params = _to_dict(params)
    props = params.get("properties") or {}
    props = _to_dict(props)

    # Flatten any nested pydantic fields
    simple = {}
    for name, meta in props.items():
        meta = _to_dict(meta)
        simple[name] = {"type": meta.get("type", "string")}
    return simple


def _load_section(
    section: str,
    template_path: Path | str = AGENT_PROMPTS,
    **kwargs,
) -> Dict[str, Any]:
    """
    Loads and renders a section of the Jinja2+YAML prompt template.

    Args:
        section (str): Section name to extract from YAML (e.g. "planner", "summarizer").
        template_path (Path|str): Path to the Jinja2 YAML prompt template.
        **kwargs: Variables to inject (user_task, tools, etc.).

    Returns:
        Dict[str, Any]: All key/value pairs under the given section.
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
) -> Dict[str, Any]:
    """
    Loads the planner prompt section, auto-injecting tool registry.

    Load and render the planner prompt configuration using Jinja2 and YAML.

    Args:
        template_path (Path|str): Path to the Jinja2 YAML prompt template.
        user_task (str): The task the user is asking for; injected into
            the prompt.
        tools (List[Dict[str, Any]]): A list of tool definitions to inject.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - system_prompt
            - select_single_tool_prompt
            - select_multi_tool_prompt
            ...
    """
    tool_registry = ToolRegistry()

    tools_for_prompt = []
    for name, spec in tool_registry.tools.items():
        s = _to_dict(spec)
        tools_for_prompt.append(
            {
                "name": name,
                "description": s.get("description", ""),
                "usage_hint": s.get("usage_hint", ""),
                "parameters": _extract_parameters(s),
            }
        )

    return _load_section(
        section="planner",
        template_path=template_path,
        user_task=user_task,
        tools=tools_for_prompt,
    )


def load_summarizer_prompt(
    log_text: str = "",
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, Any]:
    """
    Loads the summarizer prompt configuration from a YAML file.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - system_prompt
            - user_prompt
    """
    return _load_section(
        section="summarizer",
        template_path=template_path,
        log_text=log_text,
        user_task=user_task,
    )


def load_evaluator_prompt(
    task: str = "",
    response: str = "",
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, Any]:
    """
    Loads the evaluator prompt section.
    """
    return _load_section(
        section="evaluator",
        template_path=template_path,
        task=task,
        response=response,
        user_task=user_task,
    )


def load_intent_classifier_prompts(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
    """
    Loads just the task_decomposition subkey from intent_classifier.
    """
    prompts = _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
    )
    task_decomp_prompts = prompts["task_decomposition"]
    if not isinstance(task_decomp_prompts, dict):
        raise TypeError(
            f"Expected a dict for task_decomposition, got {type(task_decomp_prompts).__name__} ({task_decomp_prompts!r})"
        )
    return task_decomp_prompts


def load_describe_only_prompt(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, Any]:
    """
    Loads just the describe_only subkey from intent_classifier.
    """
    prompts = _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
    )
    desc_only_prompts = prompts["describe_only"]
    if not isinstance(desc_only_prompts, dict):
        raise TypeError(
            f"Expected a dict for describe_only, got {type(desc_only_prompts).__name__} ({desc_only_prompts!r})"
        )
    return desc_only_prompts


# # Static (global) prompt dict for classifier-only (no runtime injection needed)
# try:
#     PROMPTS = _load_section("intent_classifier")
# except Exception as e:
#     PROMPTS = {}
#     logger.error(
#         "[load_agent_prompts] Failed to load global intent_classifier prompts: %s", e
#     )
