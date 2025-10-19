"""
prompts/load_agent_prompts.py

MCP-native prompt loader for Bot0 Config Agent.
- Uses Jinja2 + YAML to render structured prompt sections.
- Injects MCP-discovered tools dynamically (no local ToolRegistry).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
from jinja2 import Environment, FileSystemLoader
import yaml
from configs.paths import AGENT_PROMPTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Core Jinja rendering helper
# ---------------------------------------------------------------------
def _load_section(
    section: str,
    template_path: Path | str = AGENT_PROMPTS,
    **kwargs,
) -> Dict[str, Any]:
    """
    Load and render a section from the YAML + Jinja template.
    Injects any provided kwargs (user_task, tools, etc.).
    """
    template_dir = str(Path(template_path).parent)
    template_name = Path(template_path).name
    logger.info(f"[load_agent_prompts] Using template: {template_path} (section='{section}')")

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    rendered = template.render(**kwargs)

    cfg = yaml.safe_load(rendered)
    return cfg.get(section, {})


# ---------------------------------------------------------------------
# Planner prompt loader (MCP-aware)
# ---------------------------------------------------------------------
def load_planner_prompts(
    user_task: str = "",
    tools: List[Dict[str, Any]] = None,
    template_path: Path | str = AGENT_PROMPTS,
    local_model: bool = False,
) -> Dict[str, Any]:
    """
    Loads the planner section with MCP-discovered tools injected.

    Args:
        user_task: User's natural-language instruction.
        tools: List of tools obtained from the MCP client (each a dict with 'name', 'description', etc.).
        template_path: Path to the YAML+Jinja template file.
        local_model: Optional flag for prompt specialization.
    """
    tools_list = tools or []
    return _load_section(
        section="planner",
        template_path=template_path,
        user_task=user_task,
        tools=tools_list,
        local_model=local_model,
    )


# ---------------------------------------------------------------------
# Other static sections remain unchanged
# ---------------------------------------------------------------------
def load_summarizer_prompts(user_task: str = "", template_path: Path | str = AGENT_PROMPTS) -> Dict[str, Any]:
    return _load_section(section="summarizer", template_path=template_path, user_task=user_task)


def load_evaluator_prompts(task: str = "", response: str = "", template_path: Path | str = AGENT_PROMPTS) -> Dict[str, Any]:
    return _load_section(section="evaluator", template_path=template_path, task=task, response=response, user_task=task)


def load_intent_classifier_prompts(user_task: str = "", template_path: Path | str = AGENT_PROMPTS) -> Dict[str, Any]:
    return _load_section(section="intent_classifier", template_path=template_path, user_task=user_task)


def load_task_decomposition_prompt(user_task: str = "", template_path: Path | str = AGENT_PROMPTS) -> Dict[str, Any]:
    prompts = _load_section(section="intent_classifier", template_path=template_path, user_task=user_task)
    return prompts.get("task_decomposition", {})


def load_describe_only_prompt(user_task: str = "", template_path: Path | str = AGENT_PROMPTS) -> Dict[str, Any]:
    prompts = _load_section(section="intent_classifier", template_path=template_path, user_task=user_task)
    return prompts.get("describe_only", {})

