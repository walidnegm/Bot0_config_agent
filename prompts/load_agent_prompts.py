"""prompts/load_agent_prompts.py"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader
import yaml
from configs.paths import AGENT_PROMPTS


def load_planner_prompts(
    template_path: Path | str = AGENT_PROMPTS,
    user_task: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """
    Load and render the planner prompt configuration using Jinja2 and YAML.

    Args:
        template_path (Path|str): Path to the Jinja2 YAML prompt template.
        user_task (str): The task the user is asking for; injected into
            the prompt.
        tools (List[Dict[str, Any]]): A list of tool definitions to inject.

    Returns:
        Dict[str, str]: A dictionary containing:
            - system_prompt
            - select_tools_prompt
            - return_json_only_prompt
            - multi_step_prompt
            - user_task_prompt
    """
    if tools is None:
        raise ValueError("tools list is required")

    template_dir = str(Path(template_path).parent)
    template_name = Path(template_path).name

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    rendered = template.render(user_task=user_task, tools=tools)
    config = yaml.safe_load(rendered)

    planner = config["planner"]

    return {
        "system_prompt": planner["system_prompt"],
        "select_tools_prompt": planner["select_tools_prompt"],
        "return_json_only_prompt": planner["return_json_only_prompt"],
        "multi_step_prompt": planner["multi_step_prompt"],
        "user_prompt": planner["user_task_prompt"].format(
            user_task=user_task
        ),  # user's task
    }


def load_summarizer_prompt(
    template_path: Path | str = AGENT_PROMPTS,
    log_text: Optional[str] = None,
) -> Dict[str, str]:
    """
    Loads the summarizer prompt configuration from a YAML file.

    Args:
        template_path (Path|str): Path to the YAML prompt file.
        log_text (str, optional): Text to inject into the user_prompt_template
            (if desired).

    Returns:
        Dict[str, str]: Dictionary with keys:
            - system_prompt
            - user_prompt_template
    """
    with open(template_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    summarizer = config.get("summarizer", {})

    system_prompt = summarizer.get("system_prompt", "")
    user_prompt_template = summarizer.get("user_prompt_template", "")

    if log_text is not None:
        user_prompt_template = user_prompt_template.format(log_text=log_text)

    return {
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
    }
