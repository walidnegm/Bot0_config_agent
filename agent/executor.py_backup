# agent/executor.py

from tools.tool_registry import ToolRegistry

class ToolExecutor:
    def __init__(self):
        self.registry = ToolRegistry()

    def execute_plan(self, plan: list, dry_run: bool = False) -> list:
        results = []

        if not isinstance(plan, list):
            return [{
                "tool": "[executor]",
                "status": "error",
                "message": f"Plan must be a list, got {type(plan).__name__}: {plan}"
            }]

        for i, step in enumerate(plan):
            print(f"[Executor] Step {i}: {step} (type: {type(step).__name__})")

            if not isinstance(step, dict):
                results.append({
                    "tool": f"[step_{i}]",
                    "status": "error",
                    "message": f"Invalid step type: expected dict, got {type(step).__name__}: {step}"
                })
                continue

            tool_name = step.get("tool")
            if not tool_name or not isinstance(tool_name, str):
                results.append({
                    "tool": f"[step_{i}]",
                    "status": "error",
                    "message": f"Missing or invalid 'tool' key in step: {step}"
                })
                continue

            params = step.get("params", {})
            if not isinstance(params, dict):
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "message": f"Invalid 'params' format: expected dict, got {type(params).__name__}"
                })
                continue

            if dry_run:
                results.append({
                    "tool": tool_name,
                    "status": "dry_run",
                    "params": params,
                    "message": "Tool not executed (dry run mode)"
                })
                continue

            try:
                tool_fn = self.registry.get_function(tool_name)
                output = tool_fn(**params)

                if not isinstance(output, dict):
                    output = {
                        "result": output,
                        "status": "ok",
                        "message": ""
                    }

                results.append({
                    "tool": tool_name,
                    "status": output.get("status", "ok"),
                    "message": output.get("message", ""),
                    "result": output.get("result", output)
                })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "message": str(e)
                })

        return results

