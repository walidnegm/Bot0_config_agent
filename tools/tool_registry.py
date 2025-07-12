import json
from pathlib import Path

class ToolRegistry:
    def __init__(self, registry_path="tools/tool_registry.json"):
        self.registry_path = Path(registry_path)
        self.tools = self._load_tools()

    def _load_tools(self):
        with open(self.registry_path, "r") as f:
            tools = json.load(f)
        print(f"[ToolRegistry] Loaded {len(tools)} tools from {self.registry_path}")
        return tools

    def get_all(self):
        return self.tools

    def get_tool(self, name):
        return next((t for t in self.tools if t["name"] == name), None)

    def get_function(self, name):
        import importlib
        try:
            module = importlib.import_module(f"tools.{name}")
            return getattr(module, "call")
        except Exception as e:
            raise RuntimeError(f"Tool '{name}' error: {e}")

