# tools/tool_registry.py

import json
import importlib
from pathlib import Path


class ToolRegistry:
    def __init__(self, registry_path="tools/tool_registry.json"):
        self.registry_path = Path(registry_path)
        self.tools = self._load_and_validate_tools()

    def _load_and_validate_tools(self):
        print(f"[ToolRegistry] 📂 Loading tools from: {self.registry_path}")
        with open(self.registry_path, "r") as f:
            raw_tools = json.load(f)

        validated = {}
        for name, spec in raw_tools.items():
            try:
                self._validate_tool(name, spec)
                validated[name] = spec
                print(f"[ToolRegistry] ✅ Validated tool '{name}'")
            except Exception as e:
                print(f"[ToolRegistry] ❌ Validation failed for tool '{name}': {e}")

        print(f"[ToolRegistry] 🎯 {len(validated)} tools validated successfully.")
        return validated

    def _validate_tool(self, name, spec):
        required_keys = ["description", "import_path", "parameters"]
        for key in required_keys:
            if key not in spec:
                raise ValueError(f"Missing required key '{key}'")

        if not isinstance(spec["parameters"], dict):
            raise TypeError("parameters must be a JSON schema object")

        import_path = spec["import_path"]
        module_path, func_name = import_path.rsplit(".", 1)

        module = importlib.import_module(module_path)
        if not hasattr(module, func_name):
            raise ImportError(f"Function '{func_name}' not found in module '{module_path}'")

    def get_all(self):
        return self.tools


    def get_function(self, name):
        try:
            tool_spec = self.tools.get(name)
            if not tool_spec:
                raise ValueError(f"Tool '{name}' not found in registry.")

            import_path = tool_spec["import_path"]
            module_path, func_name = import_path.rsplit(".", 1)

            print(f"[ToolRegistry] 🔍 Importing {func_name} from {module_path}")
            module = importlib.import_module(module_path)
            raw_function = getattr(module, func_name)

            def wrapped_function(**kwargs):
                try:
                    result = raw_function(**kwargs)
                    if isinstance(result, dict) and "result" in result and "status" in result:
                        return result
                    return {
                        "status": "ok",
                        "message": "",
                        "result": result
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "message": str(e),
                        "result": None
                    }

            return wrapped_function

        except Exception as e:
            print(f"[ToolRegistry] ❌ Failed to load function for tool '{name}': {e}")
            raise RuntimeError(f"Tool '{name}' error: {e}")


