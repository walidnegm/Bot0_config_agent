"""
tests/unit_test_loading_prompts.py

unit test: prompts/agent_prompts.yaml.j2
"""

from pathlib import Path
import pytest
import yaml
from jinja2 import Environment, FileSystemLoader
from configs.paths import AGENT_PROMPTS
from tools.tool_registry import ToolRegistry

TEMPLATE_PATH = AGENT_PROMPTS


def get_tools_for_prompt_template() -> list[dict]:
    registry = ToolRegistry()
    tools = registry.get_all()

    prompt_tools = [
        {
            "name": name,
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {}).get("properties", {}),
            "usage_hint": "",
        }
        for name, spec in tools.items()
    ]
    assert prompt_tools, "Tool registry appears empty — check test setup."
    return prompt_tools


@pytest.fixture(scope="module")
def tools():
    return get_tools_for_prompt_template()


def render_prompt_template(user_task: str, tools: list[dict]) -> dict:
    env = Environment(loader=FileSystemLoader(TEMPLATE_PATH.parent.as_posix()))
    template = env.get_template(TEMPLATE_PATH.name)
    rendered = template.render(
        planner={
            "user_task": user_task,
            "tools": tools,
        }
    )
    return yaml.safe_load(rendered)


def test_prompt_template_renders_without_error(tools):
    rendered = render_prompt_template("show me the files", tools)
    assert "planner" in rendered
    assert "system_prompt" in rendered["planner"]
    assert "user_prompt" in rendered["planner"]


def test_tool_names_rendered(tools):
    rendered = render_prompt_template("show me the files", tools)
    prompt = rendered["planner"]["select_tools_prompt"]
    assert "list_project_files" in prompt
    assert "echo_message" in prompt
    assert "root" in prompt
    assert "message" in prompt


def test_user_task_interpolated(tools):
    rendered = render_prompt_template("summarize this config", tools)
    user_prompt = rendered["planner"]["user_prompt"]
    filled = user_prompt.format(user_task="summarize this config")
    assert "summarize this config" in filled


def test_json_format_guidance_present(tools):
    rendered = render_prompt_template("something", tools)
    return_block = rendered["planner"]["return_json_only_prompt"]
    assert "Your ONLY output must be a single valid JSON array" in return_block
    assert "tool" in return_block
    assert "params" in return_block
    assert "code blocks (no ```)" in return_block or "no ```" in return_block.lower()


def test_multistep_example_present(tools):
    rendered = render_prompt_template("echo files", tools)
    block = rendered["planner"]["multi_step_prompt"]
    assert '"tool": "list_project_files"' in block
    assert '"tool": "echo_message"' in block


def test_all_tools_rendered(tools):
    rendered = render_prompt_template("render all tools", tools)
    prompt = rendered["planner"]["select_tools_prompt"]
    for tool in tools:
        assert tool["name"] in prompt
