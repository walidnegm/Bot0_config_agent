"""
tools/workbench/tool_registry.py

Registry for tool metadata and transform routing.

What this module does
---------------------
- Loads and validates tool specs from JSON (see configs.paths.TOOL_REGISTRY).
- Returns callable tool functions via their `import_path`.
- Loads transformation mappings from JSON (configs.paths.TOOL_TRANSFORMATION)
  and automatically applies the correct transform between consecutive tools.
- Validates transformed payloads against the target tool's **input Pydantic model**.

Quick start
-----------
from bot0_config_agent.tools.configs.tool_registry import ToolRegistry
from bot0_config_agent.tools.configs.tool_models import ToolOutput
from bot0_config_agent.agent_models.step_status import StepStatus

registry = ToolRegistry()

# Inspect tools (ToolSpec objects)
for name, spec in registry.get_all().items():
    print(name, "→", spec.import_path, "|", spec.description)

# Get and call a tool function
list_files = registry.get_function("list_project_files")
out = list_files(dir=".", exclude=[".venv", "__pycache__"], include=[".py", ".md"])
print(out)  # expected envelope: {"status": ..., "result": ..., "message": ...}

# Convert one tool's output to the next tool's input (auto-looks up transform)
tool_output = ToolOutput(status=StepStatus.SUCCESS, message="", result=out.get("result"))
next_input_model = registry.match_and_convert_output(
    output=tool_output,
    target_tool="summarize_files",
    source_tool="list_project_files",  # enables (source,target) transform lookup
)
# next_input_model is a validated Pydantic model instance for the target tool input.
"""

import json
import logging
import importlib
from pathlib import Path
from typing import Any, cast, Callable, Dict, Optional, Tuple
from pydantic import BaseModel, ValidationError
from bot0_config_agent.agent_models.step_status import StepStatus
from bot0_config_agent.tools.configs.tool_models import (
    ToolSpec,
    ToolOutput,
    ToolTransformation,
)
from bot0_config_agent.configs.paths import (
    TOOL_REGISTRY,
    TOOL_TRANSFORMATION,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(
        self, registry_path=TOOL_REGISTRY, transformation_path=TOOL_TRANSFORMATION
    ):
        self.registry_path = Path(registry_path)
        self.transformation_path = (
            Path(transformation_path) if transformation_path else None
        )

        # Load tool specs
        self.tools: Dict[str, ToolSpec] = self._load_and_validate_tools()

        # Load transforms ONCE into a lookup map: (source_tool, target_tool) -> ToolTransformation
        self._transform_map: Dict[Tuple[str, str], ToolTransformation] = (
            self._load_transformation()
        )

    def get_all(self) -> Dict[str, ToolSpec]:
        """Get all tools"""
        return self.tools

    def get_function(self, name: str):
        """
        Return the callable tool function. Optionally resolves the output model (if present).
        """
        spec = self.tools.get(name)
        if not spec:
            raise ValueError(f"Unknown tool: {name}")

        import_path = spec.import_path
        if not import_path or "." not in import_path:
            raise ValueError(f"Invalid import_path for tool '{name}': {import_path}")

        mod_path, fn_name = import_path.rsplit(".", 1)
        module = importlib.import_module(mod_path)
        return getattr(module, fn_name)

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

        else:
            raise TypeError(f"Unsupported tool spec type: {type(spec).__name__}")

    def match_and_convert_output(
        self,
        *,
        output: ToolOutput,
        target_tool: str,
        source_tool: Optional[str] = None,
        transform_fn: Optional[Callable[[Any], Any]] = None,
    ) -> BaseModel:
        """
        Match a tool's output to the target tool's input model, applying a transformation
        if needed.

        This helper:
        1. Checks that the source tool output status is SUCCESS.
        2. Loads the target tool's registered input model from the registry.
        3. If no explicit `transform_fn` is provided, tries to get one from the registry
            using `source_tool` and `target_tool`.
        4. If no transform is found but the models match, performs direct
            Pydantic validation.
        5. If a transform is found, applies it to `output.result`.
        6. Validates the transformed payload against the target tool's input model.

        Args:
            output: ToolOutput instance from the source tool.
            target_tool: Name of the tool whose input model we want to validate against.
            source_tool: Name of the tool that generated this output (used for
                auto-matching transformation).
            transform_fn: Optional callable to transform the payload before validation.

        Returns:
            An instance of the target tool's input model.

        Raises:
            ValueError: For unknown tools, invalid transformation, or status != SUCCESS.
            ValidationError: If payload does not match the target input model schema.
        """
        # 1. Ensure the source tool succeeded before conversion
        if output.status != StepStatus.SUCCESS:
            raise ValueError(f"Cannot process output with status {output.status}")

        # 2. Retrieve the target tool spec from registry
        spec = self.tools.get(target_tool)
        if not spec:
            raise ValueError(f"Unknown target tool: {target_tool}")

        # 3. Load the Pydantic input model for the target tool
        input_model = self._load_model(spec.input_model)
        if not input_model:
            raise ValueError(
                f"Input model for tool '{target_tool}' not found or invalid"
            )

        # 4. Auto-lookup transform function if none is passed (straight convert)
        if transform_fn is None and source_tool is not None:
            transform = self._transform_map.get((source_tool, target_tool))
            if transform and transform.transform_fn:
                transform_fn = self._resolve_callable(transform.transform_fn)

        # 5. Start with raw result payload
        payload = output.result

        # 6. If a transform function exists, apply it
        if transform_fn:
            try:
                payload = transform_fn(payload)
            except Exception as e:
                raise ValueError(f"Transformation failed: {e}")

        # 7. Convert payload to dict if it's a Pydantic model, &
        # Validate payload with the target input model
        try:
            if isinstance(payload, BaseModel):
                payload = payload.model_dump()
            data = payload if isinstance(payload, dict) else {"result": payload}
            return input_model.model_validate(data)
        except ValidationError as e:
            raise ValueError(
                f"Output does not match {input_model.__name__} for tool '{target_tool}': {e}"
            )

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

    def _load_model(self, dotted: str | None) -> type[BaseModel] | None:
        """
        Load a Pydantic model class given its name or dotted path.
        Supports both input_model and output_model.

        - If dotted path: "pkg.module.ClassName" → import directly
        - If bare name:   "ClassName" → assume tools.workbench.tool_models
        - If None:        returns None

        Example:
        >>> spec.input_model = "ReadFilesInput"
        >>> input_model = self._load_model(spec.input_model)
        >>> # input_model is <class 'tools.workbench.tool_models.ReadFilesInput'>
        """
        if not dotted:
            return None
        if not isinstance(dotted, str):  # already a class
            return dotted
        try:
            if "." in dotted:
                module_path, cls_name = dotted.rsplit(".", 1)
                module = importlib.import_module(module_path)
            else:
                module = importlib.import_module("tools.workbench.tool_models")
                cls_name = dotted
            model_cls = getattr(module, cls_name)
            if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
                raise TypeError(f"{dotted} is not a Pydantic BaseModel subclass")
            return model_cls
        except Exception as e:
            logger.error(f"[ToolRegistry] Failed to resolve model '{dotted}': {e}")
            return None

    def _load_transformation(self) -> Dict[Tuple[str, str], ToolTransformation]:
        """Return a map keyed by (source_tool, target_tool)."""
        if not self.transformation_path or not self.transformation_path.exists():
            logger.info(f"[ToolRegistry] No transforms at {self.transformation_path}")
            return {}

        try:
            data = json.loads(self.transformation_path.read_text())
            out: Dict[Tuple[str, str], ToolTransformation] = {}
            for item in data:
                tt = ToolTransformation.model_validate(item)
                out[(tt.source_tool, tt.target_tool)] = tt
            logger.info(f"[ToolRegistry] Loaded {len(out)} tool transformation.")
            return out
        except Exception as e:
            logger.error(f"[ToolRegistry] Failed to load transformation: {e}")
            return {}

    def _load_transform_fn(self, dotted: str) -> Callable[[Any], Any] | None:
        """
        Resolve a dotted path to a callable transform function.

        This turns a string like "tools.transformation.dir_to_files" into a Python
        callable using Python's import system. The function is intentionally tolerant:
        it logs a single descriptive error and returns None on any failure so callers
        can decide how to proceed.

        Behavior:
        - Requires an absolute dotted path containing at least one dot.
        - Imports the target module and fetches the named attribute.
        - Verifies the resolved attribute is callable.
        - Returns the callable on success; otherwise logs and returns None.

        Args:
            dotted: Absolute dotted path to a function (e.g., "pkg.module.func").

        Returns:
            Callable[[Any], Any] | None: The resolved function, or None if import/lookup
            fails or the object is not callable.

        Notes:
            - Uses str.rpartition('.') to avoid ValueError from str.rsplit and to
            pre-validate input.
            - For strict behavior, replace the early return and the except block with
            `raise` to propagate errors to the caller.

        Examples:
            >>> fn = self._load_transform_fn("tools.transformation.dir_to_files")
            >>> if fn:
            ...     out = fn(prev_result)
        """
        dotted = (dotted or "").strip()
        if not dotted or "." not in dotted:
            logger.error("[ToolRegistry] Invalid transform path: %r", dotted)
            return None

        module_path, _, fn_name = dotted.rpartition(
            "."
        )  # returns before, separator, after split
        try:
            obj = getattr(importlib.import_module(module_path), fn_name)
            if not callable(obj):
                raise TypeError(
                    f"Resolved '{dotted}' to non-callable {type(obj).__name__}"
                )
            return cast(Callable[[Any], Any], obj)
        except Exception as e:
            logger.error(
                "[ToolRegistry] Failed loading transform '%s' (%s: %s)",
                dotted,
                e.__class__.__name__,
                e,
            )
            return None

    def _resolve_callable(self, dotted: str) -> Callable:
        mod, fn = dotted.rsplit(".", 1)
        module = importlib.import_module(mod)
        obj = getattr(module, fn)
        if not callable(obj):
            raise TypeError(f"Transform '{dotted}' is not callable")
        return obj
