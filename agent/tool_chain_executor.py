"""agent/tool_steps_executor.py"""

"""agent/tool_steps_executor.py"""

import re
import copy
from typing import Any, Dict, List
from tools.tool_registry import ToolRegistry


class ToolStepsExecutor:
    """
    Minimal FSM-like executor for sequential tool chains (steps).
    - Treats each plan step as a stage (step_0, step_1, ...)
    - Stores each step's full output in context
    - Resolves <step_n.result.key[index]> references in params
    """

    def __init__(self):
        self.registry = ToolRegistry()  # ✅ instantiate
        self.context: Dict[str, Dict[str, Any]] = {}  # step_n -> full tool output

    def run_plan(
        self, plan: List[Dict[str, Any]], dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        results = []

        for i, step in enumerate(plan):
            step_id = f"step_{i}"
            tool_name = step.get("tool")
            params = step.get("params", {})

            if not tool_name or not isinstance(params, dict):
                results.append(
                    {
                        "tool": tool_name or f"[{step_id}]",
                        "status": "error",
                        "message": f"Invalid step format at index {i}: {step}",
                    }
                )
                continue

            # ✅ Resolve <step_n> references
            resolved_params = self._resolve_references(params)

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

                # Ensure standard format
                if not isinstance(output, dict):
                    output = {
                        "status": "ok",
                        "result": output,
                        "message": "",
                    }  # ✅ fixed typo

                result_entry = {
                    "tool": tool_name,
                    "status": output.get("status", "ok"),
                    "message": output.get("message", ""),
                    "result": output.get("result", output),
                }

                results.append(result_entry)
                self.context[step_id] = result_entry  # Save for future reference

            except Exception as e:
                results.append(
                    {"tool": tool_name, "status": "error", "message": str(e)}
                )

        return results

    def _resolve_references(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace <step_n.result.key[index]> placeholders with actual values from context.
        Defaults to 'result' if no explicit root given.
        """
        resolved = copy.deepcopy(params)

        for k, v in resolved.items():
            if isinstance(v, str) and v.startswith("<") and v.endswith(">"):
                placeholder = v[1:-1]  # strip < >
                parts = placeholder.split(".", 1)

                # Step reference (i.e., step_0)
                step_id = parts[0]
                if step_id not in self.context:
                    resolved[k] = None
                    continue

                data = self.context[step_id]

                # Default to "result" if no explicit path
                if len(parts) == 1:
                    data = data.get("result")
                else:
                    path = parts[1]
                    if not path.startswith("result") and not path.startswith("message"):
                        path = f"result.{path}"
                    data = self._extract_path(data, path)

                resolved[k] = data
        return resolved

    def _extract_path(self, source: Any, path: str) -> Any:
        """
        Traverse a dict/list structure using dot notation and [index] syntax.
        Example: result.files[0] -> source['result']['files'][0]
        """
        parts = re.split(r"\.(?![^\[]*\])", path)  # split on '.' not inside []
        val = source
        for part in parts:
            if "[" in part and part.endswith("]"):
                field, idx = part[:-1].split("[")
                if field:
                    val = val.get(field) if isinstance(val, dict) else None
                val = val[int(idx)] if isinstance(val, (list, tuple)) else None
            else:
                val = val.get(part) if isinstance(val, dict) else None
            if val is None:
                break
        return val
