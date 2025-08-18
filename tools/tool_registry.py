"""
tools/tool_registry.py

Usage:
    from tools.tool_registry import ToolRegistry
    registry = ToolRegistry()
    print(registry.has_tool("llm_response_async"))
    fn = registry.get_function("llm_response_async", api_model_name="gpt-4.1-mini")
    out = fn(prompt="Hello")
    print(out)
"""
from __future__ import annotations

import json
import logging
import importlib
from pathlib import Path
from typing import Callable, Any, Dict, Optional

from pydantic import BaseModel, ValidationError

from tools.tool_models import ToolSpec
from configs.paths import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, registry_path: Path | str = TOOL_REGISTRY):
        self.registry_path = Path(registry_path)
        self.tools: Dict[str, ToolSpec] = self._load_and_validate_tools()

    # ---------- Public API ----------

    def get_all(self) -> Dict[str, ToolSpec]:
        return self.tools

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def get_function(self, name: str, **extra_kwargs) -> Callable[..., Any]:
        """
        Return the callable tool function with optional extra kwargs injected
        (e.g., model names). Also resolves the output model if present.

        Raises:
            ValueError if the tool is unknown or cannot be imported.
        """
        spec = self.tools.get(name)
        if not spec:
            raise ValueError(f"Unknown tool: {name}")

        # Resolve the tool function
        import_path = spec.import_path
        if not import_path or "." not in import_path:
            raise ValueError(f"Invalid import_path for tool '{name}': {import_path!r}")

        try:
            mod_path, fn_name = import_path.rsplit(".", 1)
            module = importlib.import_module(mod_path)
            tool_fn = getattr(module, fn_name)
        except (ImportError, AttributeError) as e:
            logger.error(
                "Failed to import tool '%s' from %s: %s", name, import_path, e
            )
            raise ValueError(
                f"Tool '{name}' import failed from '{import_path}': {e}"
            ) from e

        # Wrap function to include extra kwargs (e.g., local_model_name/api_model_name)
        def wrapped_tool_fn(*args, **kwargs):
            combined_kwargs = {**kwargs, **extra_kwargs}
            return tool_fn(*args, **combined_kwargs)

        # Optionally resolve output model (for future validation/use)
        out_model_name: Optional[str] = spec.output_model
        out_model_cls = self._load_output_model(out_model_name)
        if out_model_cls:
            logger.debug(
                "[ToolRegistry] Resolved output model for '%s': %s",
                name,
                out_model_name,
            )

        return wrapped_tool_fn

    def get_param_keys(self, tool_name: str) -> set[str]:
        """
        Return the set of valid parameter keys for the given tool.
        """
        spec = self.tools.get(tool_name)
        if spec is None:
            raise KeyError(f"Unknown tool: {tool_name}")
        params = spec.parameters
        if not isinstance(params, dict):
            raise TypeError(
                f"'parameters' on ToolSpec must be dict; got {type(params).__name__}"
            )
        props = params.get("properties", {})
        if not isinstance(props, dict):
            raise ValueError("'parameters.properties' missing or not a dict.")
        return set(props.keys())

    # ---------- Internals ----------

    def _load_and_validate_tools(self) -> Dict[str, ToolSpec]:
        if not self.registry_path.exists():
            logger.error("Tool registry file not found: %s", self.registry_path)
            raise FileNotFoundError(f"Tool registry file not found: {self.registry_path}")

        try:
            text = self.registry_path.read_text(encoding="utf-8")
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in tool registry %s: %s", self.registry_path, e)
            raise

        if not isinstance(data, dict):
            raise ValueError(
                f"Tool registry JSON must be an object at top level; got {type(data).__name__}"
            )

        validated: Dict[str, ToolSpec] = {}
        for name, raw in data.items():
            try:
                tool = ToolSpec.model_validate(raw)

                # Import-time check (fail-fast here so unknown tools don’t appear “registered”)
                module_path, func_name = tool.import_path.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                if not hasattr(mod, func_name):
                    raise ImportError(f"{func_name} not found in {module_path}")

                validated[name] = tool
                logger.info("[ToolRegistry] Registered tool: %s", name)

            except (ValidationError, ImportError, ValueError) as e:
                logger.warning("Failed to load tool '%s': %s", name, e)

        if not validated:
            logger.warning(
                "No valid tools loaded from registry %s. "
                "Check JSON content and tool import paths.",
                self.registry_path,
            )
        else:
            logger.info(
                "[ToolRegistry] Loaded %d tool(s): %s",
                len(validated),
                ", ".join(sorted(validated.keys())),
            )
        return validated

    def _load_output_model(self, dotted: Optional[str]) -> Optional[type[BaseModel]]:
        """
        Load a Pydantic output model class given its name or dotted path.
        Returns None if not resolvable or not a BaseModel subclass.
        """
        if not dotted:
            return None
        if not isinstance(dotted, str):
            # Already a class/type (defensive)
            return dotted  # type: ignore[return-value]

        try:
            if "." in dotted:
                module_path, cls_name = dotted.rsplit(".", 1)
                module = importlib.import_module(module_path)
            else:
                module = importlib.import_module("tools.tool_models")
                cls_name = dotted

            model_cls = getattr(module, cls_name)
            if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
                logger.warning(
                    "Resolved '%s' but it is not a Pydantic BaseModel subclass.", dotted
                )
                return None
            return model_cls
        except Exception as e:
            logger.error("[ToolRegistry] Failed to resolve model '%s': %s", dotted, e)
            return None

