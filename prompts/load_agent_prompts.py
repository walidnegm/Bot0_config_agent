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
        user_task (str): The task the user is asking for; injected into the prompt.
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
