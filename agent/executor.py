from tools.tool_registry import ToolRegistry

class ToolExecutor:
    def __init__(self):
        self.registry = ToolRegistry()

    def execute_plan(self, plan: list, dry_run: bool = False) -> list:
        results = []

        for step in plan:
            tool_name = step["tool"]
            args = step.get("args", {})

            if dry_run:
                results.append({
                    "tool": tool_name,
                    "status": "dry_run",
                    "args": args,
                    "message": "Tool not executed (dry run mode)"
                })
                continue

            try:
                tool_fn = self.registry.get_function(tool_name)
                output = tool_fn(args)
                results.append({
                    "tool": tool_name,
                    "status": output.get("status", "unknown"),
                    "message": output.get("message", ""),
                    "result": output
                })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "message": str(e)
                })

        return results

