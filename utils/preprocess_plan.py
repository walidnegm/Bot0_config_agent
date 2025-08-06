""" "
utils/preprocess_plan.py
helper function to convert LLM's simpler response to full format
and validate with pydantic.
"""

from typing import Union
from agent_models.llm_response_models import ToolChain, ToolCall

# Maps tool name -> default path inside `.result`
TOOL_OUTPUT_SCHEMA = {
    "list_project_files": "files[0]",
    "find_urls": "urls[0]",
    "extract_text": "text",
    # Add more tools and default result paths here
}


def preprocess_plan(plan: Union[ToolChain, list[dict]]) -> ToolChain:
    """
    Convert simple 'step_n' references in ToolChain params into full
    '<step_n.result.key[index]>' references based on TOOL_OUTPUT_SCHEMA.

    This function is used in the safer step_n + mapping approach, where the
    Planner (LLM) only outputs step references (e.g., "step_0") instead of
    full <step_n.result.key[index]> paths. The mapping layer then fills in
    the correct .result path based on known tool output schemas.

    Args:
        plan (ToolChain | list[dict]):
            The tool chain to preprocess. Can be:
            - A validated ToolChain model
            - A raw list of dicts from the LLM (will be coerced into ToolChain)

    Returns:
        ToolChain:
            A new ToolChain with all 'step_n' references replaced by
            '<step_n.result.key[index]>' strings according to TOOL_OUTPUT_SCHEMA.

    Raises:
        TypeError: If the input is not a ToolChain or list of dicts.
        ValueError: If a tool has no mapping in TOOL_OUTPUT_SCHEMA for a
            referenced parameter.

    Example:
        >>> raw_plan = [
        ...     {"tool": "list_project_files", "params": {"root": "."}},
        ...     {"tool": "read_file", "params": {"path": "step_0"}}
        ... ]
        >>> resolved_plan = preprocess_plan(raw_plan)
        >>> print(resolved_plan.model_dump())
        {
            "steps": [
                {"tool": "list_project_files", "params": {"root": "."}},
                {"tool": "read_file", "params": {"path": "<step_0.result.files[0]>"}}
            ]
        }

    Expected Output:
        - All 'step_n' values in parameters are transformed into
          '<step_n.result.<schema_path>>' format.
        - Steps without 'step_n' references are left unchanged.
    """
    # If raw list, validate/convert to ToolChain
    if isinstance(plan, list):
        tool_chain = ToolChain(steps=[ToolCall(**step) for step in plan])
    elif isinstance(plan, ToolChain):
        tool_chain = plan
    else:
        raise TypeError("Plan must be a ToolChain or a list of dicts.")

    processed_steps = []
    for step in tool_chain.steps:
        tool_name = step.tool
        params = step.params.copy()
        new_params = {}

        for k, v in params.items():
            if isinstance(v, str) and v.startswith("step_"):
                ref_step = v
                default_path = TOOL_OUTPUT_SCHEMA.get(tool_name)
                if not default_path:
                    raise ValueError(
                        f"No default output schema for tool '{tool_name}' "
                        f"while mapping param '{k}' in step '{tool_name}'."
                    )
                new_params[k] = f"<{ref_step}.result.{default_path}>"
            else:
                new_params[k] = v

        processed_steps.append(ToolCall(tool=tool_name, params=new_params))

    return ToolChain(steps=processed_steps)
