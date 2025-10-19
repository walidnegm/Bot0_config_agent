"""
tools/tool_registry.py
Loads tool specifications from a JSON file and provides access to their
callable functions.
"""
import json
import logging
import importlib
from pathlib import Path
from typing import Callable, Any, Dict, Optional, List

from pydantic import BaseModel, ValidationError
from tools.tool_models import ToolSpec
from configs.paths import TOOL_REGISTRY

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self, registry_path: Path | str = TOOL_REGISTRY):
        self.registry_path = Path(registry_path)
        self.tools: Dict[str, ToolSpec] = self._load_and_validate_tools()


    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns a list of tool specifications for use in prompts."""
        tool_list = []
        for name, spec in self.tools.items():
            params = {
                prop: {"type": details.get("type", "any")}
                for prop, details in spec.parameters.get("properties", {}).items()
            }
            tool_list.append({
                "name": name,
                "description": spec.description,
                "parameters": params,
            })
        return tool_list

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def get_function(self, name: str, **extra_kwargs) -> Callable[..., Any]:
        """
        Return the callable tool function with optional extra kwargs injected.
        """
        spec = self.tools.get(name)
        if not spec:
            raise ValueError(f"Unknown tool: {name}")

        try:
            module_path, func_name = spec.import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            tool_fn = getattr(module, func_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import tool '{name}' from '{spec.import_path}': {e}")
            raise ValueError(f"Could not load tool: {name}") from e

        if not extra_kwargs:
            return tool_fn

        def wrapped_tool_fn(*args, **kwargs):
            combined_kwargs = {**kwargs, **extra_kwargs}
            return tool_fn(*args, **combined_kwargs)

        return wrapped_tool_fn

    def get_param_keys(self, tool_name: str) -> List[str]:
        """Get a list of valid parameter keys for a given tool."""
        spec = self.tools.get(tool_name)
        if not spec or not spec.parameters:
            return []
        return list(spec.parameters.get("properties", {}).keys())

    def _load_and_validate_tools(self) -> Dict[str, ToolSpec]:
        """Loads tool definitions from the JSON registry file and validates imports."""
        if not self.registry_path.exists():
            logger.error(f"[ToolRegistry] Tool registry file not found at: {self.registry_path}")
            return {}
        
        with open(self.registry_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"[ToolRegistry] Failed to parse tool_registry.json: {e}")
                return {}

        validated_tools = {}
        for name, spec_dict in data.items():
            try:
                # Validate Pydantic model
                tool_spec = ToolSpec(**spec_dict)
                # Validate import path
                try:
                    module_path, func_name = tool_spec.import_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    if not hasattr(module, func_name):
                        logger.warning(f"[ToolRegistry] Tool '{name}' import_path '{tool_spec.import_path}' is invalid: {func_name} not found")
                        continue
                except (ImportError, ValueError) as e:
                    logger.warning(f"[ToolRegistry] Failed to validate import for tool '{name}' from '{tool_spec.import_path}': {e}")
                    continue
                validated_tools[name] = tool_spec
                logger.info(f"[ToolRegistry] Registered tool: {name}")
            except ValidationError as e:
                logger.warning(f"[ToolRegistry] Failed to validate tool '{name}': {e}")
            except Exception as e:
                logger.error(f"[ToolRegistry] Unexpected error loading tool '{name}': {e}")
        
        logger.info(f"[ToolRegistry] Loaded {len(validated_tools)} tool(s): {', '.join(sorted(validated_tools.keys()))}")
        return validated_tools
