"""
tools/tool_registry.py

Example Usage:

from tools.tool_registry import ToolRegistry

# Initialize registry (loads and validates JSON file)
registry = ToolRegistry()

# Get all validated tools
tools = registry.get_all()

for tool_name, spec in tools.items():
    print(f"Tool: {tool_name}")
    print(f"  Description: {spec['description']}")
    print(f"  Import Path: {spec['import_path']}")
    print(f"  Parameters Schema: {spec['parameters']}")
    print("-" * 40)


Example Usage:

from tools.tool_registry import ToolRegistry

# Initialize registry
registry = ToolRegistry()

# Choose a tool name exactly as in tool_registry.json
tool_name = "list_project_files"

# Get the wrapped tool function
list_files = registry.get_function(tool_name)

# Call it with parameters defined in the JSON schema
result = list_files(root=".", exclude=[".venv", "__pycache__"], include=[".py", ".md"])

# The wrapped function always returns a dict with keys: status, message, result
print(result)
"""

import json
import logging
import importlib
from pathlib import Path
from typing import Callable, Any, Dict
from pydantic import BaseModel, ValidationError
from agent_models.step_status import StepStatus
from tools.tool_models import ToolSpec
from configs.paths import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, registry_path=TOOL_REGISTRY):
        self.registry_path = Path(registry_path)
        self.tools: Dict[str, ToolSpec] = self._load_and_validate_tools()

    def get_all(self) -> Dict[str, ToolSpec]:
        return self.tools

    def get_function(self, name: str):
        """
        Return the callable tool function. Optionally resolves the output model (if present).
        """
        spec = self.tools.get(name)
        if not spec:
            raise ValueError(f"Unknown tool: {name}")

        # Resolve the tool function (unchanged logic you already have)
        import_path = spec.import_path
        if not import_path or "." not in import_path:
            raise ValueError(f"Invalid import_path for tool '{name}': {import_path}")
        mod_path, fn_name = import_path.rsplit(".", 1)
        module = importlib.import_module(mod_path)
        tool_fn = getattr(module, fn_name)

        # Resolve output model *if* provided; don't crash if it's missing or unresolvable
        out_model_name = spec.output_model
        out_model_cls = self._load_output_model(out_model_name)
        # If you actually need to use it elsewhere, keep a map:
        # self.output_models[name] = out_model_cls

        return tool_fn

    def get_param_keys(self, tool_name: str) -> set[str]:
        """
        Return the set of valid parameter keys for the given tool.
        Assumes 'parameters' is always a JSON Schema dict.
        Raises TypeError if it's not a dict.
        """
        spec = self.tools.get(tool_name)
        if spec is None:
            raise KeyError(f"Unknown tool: {tool_name}")

        if hasattr(spec, "parameters"):  # ToolSpec/Pydantic path
            params = spec.parameters
            if not isinstance(params, dict):
                raise TypeError(
                    f"'parameters' on ToolSpec must be dict; got {type(params).__name__}"
                )
            props = params.get("properties", {})
            if not isinstance(props, dict):
                raise ValueError("'parameters.properties' missing or not a dict.")
            return set(props.keys())

        if isinstance(spec, dict):  # Raw dict path
            props = spec.get("parameters", {}).get("properties", {})
            if not isinstance(props, dict):
                raise ValueError("'parameters.properties' missing or not a dict.")
            return set(props.keys())

        raise TypeError(f"Unsupported tool spec type: {type(spec).__name__}")

    def _load_and_validate_tools(self) -> Dict[str, ToolSpec]:
        data = json.loads(Path(self.registry_path).read_text())
        validated = {}
        for name, spec in data.items():
            tool = ToolSpec.model_validate(spec)
            # import-time check
            module_path, func_name = tool.import_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            if not hasattr(mod, func_name):
                raise ImportError(f"{func_name} not in {module_path}")
            validated[name] = tool
        return validated

    def _load_output_model(self, dotted: str | None) -> type[BaseModel] | None:
        """
        Load a Pydantic output model class given its name or dotted path.

        - If dotted path: "pkg.module.ClassName" → import directly
        - If bare name:   "ClassName" → assume tools.tool_models
        - If None:        returns None
        """
        if not dotted:
            return None

        # If it's already a class object (defensive), return as-is
        if not isinstance(dotted, str):
            return dotted

        try:
            if "." in dotted:
                module_path, cls_name = dotted.rsplit(".", 1)
                module = importlib.import_module(module_path)
            else:
                # Default to tools.tool_models for bare names
                module = importlib.import_module("tools.tool_models")
                cls_name = dotted

            return getattr(module, cls_name)
        except Exception as e:
            logger.error(f"[ToolRegistry] Failed to resolve model '{dotted}': {e}")
            return None  # Be tolerant — calling code should handle None
