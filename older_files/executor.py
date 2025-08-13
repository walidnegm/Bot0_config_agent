"""
# todo: this module is deprecated. will delete soon.

ToolExecutor: Executes a sequence of tool calls from a plan.

- Executes tools from the ToolRegistry.
- Does NOT load or manage any LLM models or configs.
- Does NOT fallback to raw LLM chat completions (that is handled centrally).

* removed the model calling part - ToolExecutor should be executing tools only.
"""

from typing import List, Dict, Any
from tools.tool_registry import ToolRegistry


class ToolExecutor:
    def __init__(self):
        self.registry = ToolRegistry()

    def execute_plan(self, plan: List[dict], dry_run: bool = False) -> List[dict]:
        """
        Execute a list of tool calls (the plan), in order.

        Args:
            plan (List[dict]): Sequence of tool call dicts with 'tool' and 'params'.
            dry_run (bool): If True, tools are not executed (for testing/debug).

        Returns:
            List[dict]: List of tool results (one per step).
        """
        results = []
        prev_output = None
        step_outputs = {}

        if not isinstance(plan, list):
            return [
                {
                    "tool": "[executor]",
                    "status": "error",
                    "message": f"Plan must be a list, got {type(plan).__name__}: {plan}",
                }
            ]

        for i, step in enumerate(plan):
            print(f"[Executor] Step {i}: {step} (type: {type(step).__name__})")

            if not isinstance(step, dict):
                results.append(
                    {
                        "tool": f"[step_{i}]",
                        "status": "error",
                        "message": f"Invalid step type: expected dict, got {type(step).__name__}: {step}",
                    }
                )
                continue

            tool_name = step.get("tool")
            if not tool_name or not isinstance(tool_name, str):
                results.append(
                    {
                        "tool": f"[step_{i}]",
                        "status": "error",
                        "message": f"Missing or invalid 'tool' key in step: {step}",
                    }
                )
                continue

            params = step.get("params", {})
            if not isinstance(params, dict):
                results.append(
                    {
                        "tool": tool_name,
                        "status": "error",
                        "message": f"Invalid 'params' format: expected dict, got {type(params).__name__}",
                    }
                )
                continue

            if tool_name not in self.registry.tools:
                results.append(
                    {
                        "tool": tool_name,
                        "status": "error",
                        "message": f"Invalid tool: {tool_name} not found in registry.",
                    }
                )
                continue

            # Handle placeholders: <prev_output> and <step_n>
            resolved_params = {}
            for k, v in params.items():
                if isinstance(v, str):
                    # <prev_output> placeholder
                    if "<prev_output>" in v:
                        resolved_val = (
                            str(prev_output.get("message", prev_output))
                            if isinstance(prev_output, dict)
                            else str(prev_output)
                        )
                        v = v.replace("<prev_output>", resolved_val)

                    # <step_n> placeholders
                    for ref_idx in range(i):
                        ref_token = f"<step_{ref_idx}>"
                        if ref_token in v and f"step_{ref_idx}" in step_outputs:
                            v = v.replace(
                                ref_token, str(step_outputs[f"step_{ref_idx}"])
                            )
                resolved_params[k] = v

            if dry_run:
                results.append(
                    {
                        "tool": tool_name,
                        "status": "dry_run",
                        "params": resolved_params,
                        "message": "Tool not executed (dry run mode)",
                    }
                )
                continue

            try:
                tool_fn = self.registry.get_function(tool_name)
                output = tool_fn(**resolved_params)

                if not isinstance(output, dict):
                    output = {"result": output, "status": "ok", "message": ""}

                if "status" not in output:
                    output["status"] = "ok"

                print(
                    f"[Executor] ✅ Tool '{tool_name}' succeeded with status: {output['status']}"
                )
                result = {
                    "tool": tool_name,
                    "status": output.get("status", "ok"),
                    "message": output.get("message", ""),
                    "result": output.get("result", output),
                }

                results.append(result)
                prev_output = result["result"]
                step_outputs[f"step_{i}"] = prev_output

            except Exception as e:
                print(f"[Executor] ❌ Tool {tool_name} failed: {e}")
                results.append(
                    {
                        "tool": tool_name,
                        "status": "error",
                        "message": str(e),
                    }
                )

        return results
