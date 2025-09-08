"""
tools/workbench/tool_registry.py

* Core code for registry for tool metadata and transform routing.

What this module does
---------------------
- Loads and validates tool specs either from:
    (A) a DIRECTORY that contains *.tool.json files (optionally controlled
        by _manifest.yaml), OR
    (B) a legacy single JSON file (mapping name -> spec).
- Returns callable tool functions via their `import_path`.
- Loads transformation mappings from JSON (configs.paths.TOOL_TRANSFORMATION)
  and automatically applies the correct transform between consecutive tools.
- Validates transformed payloads against the target tool's **input Pydantic model**.

Quick start
-----------
from bot0_config_agent.tools.configs.tool_registry import ToolRegistry
from bot0_config_agent.tools.configs.tool_models import ToolOutput
from bot0_config_agent.agent_models.step_status import StepStatus

registry = ToolRegistry()  # respects configs.paths.TOOL_REGISTRY being a dir or a file


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
from typing import Any, cast, Callable, Dict, Iterable, Optional, Tuple
from pydantic import BaseModel, ValidationError
from bot0_config_agent.agent_models.step_status import StepStatus
from bot0_config_agent.tools.configs.tool_models import (
    ToolSpec,
    ToolOutput,
    ToolTransformation,
)
from bot0_config_agent.tools.configs.tool_registry_loader import load_tool_registry

from bot0_config_agent.configs.paths import (
    TOOL_REGISTRY_JSON_FILE,
    TOOL_TRANSFORMATION_JSON_FILE,
    TOOLS_REGISTRY_DIR,
)

logger = logging.getLogger(__name__)


DEFAULT_TOOL_MODEL_MODULE_PATHS = ["bot0_config_agent.tools.configs.tool_models"]


class ToolRegistry:
    """
    High-level façade around tool specifications and inter-tool transformations.

    The registry loads from directory-based layout:
       - TOOL_REGISTRY points to a directory.
       - Directory contains one or more `*.tool.json` files.
       - Optional `_manifest.yaml` controls which files to include and per-file
       overrides.
       - Use `tool_registry_loader.load_tool_registry()` underneath.
    """

    def __init__(
        self,
        # registry_path: Path | str = TOOL_REGISTRY_JSON_FILE,
        registry_dir: Path | str = TOOLS_REGISTRY_DIR,
        transformation_path: Path | str | None = TOOL_TRANSFORMATION_JSON_FILE,
        *,
        import_sanity_check: bool = True,
    ):
        self.registry_path = Path(registry_dir).resolve()
        self.transformation_path = (
            Path(transformation_path) if transformation_path else None
        )
        self.import_sanity_check = import_sanity_check

        # Load tool specs
        self.tools: Dict[str, ToolSpec] = self._load_and_validate_tools()

        # Load transforms ONCE into a lookup map: (source_tool, target_tool) -> ToolTransformation
        self._transform_map: Dict[Tuple[str, str], ToolTransformation] = (
            self._load_transformation()
        )

    # ------------------------------- Public API -------------------------------

    def get_all(self) -> Dict[str, ToolSpec]:
        """Get a mapping of tool name -> ToolSpec."""
        return self.tools

    def get_function(self, name: str):
        """
        Return the callable tool function. Optionally resolves the output model
        (if present).
        """
        spec = self.tools.get(name)
        if not spec:
            raise ValueError(f"Unknown tool: {name}")

        import_path = spec.import_path
        if not import_path or "." not in import_path:
            raise ValueError(f"Invalid import_path for tool '{name}': {import_path}")

        mod_path, fn_name = import_path.rsplit(
            ".", 1
        )  # example: ...tools.tool_scripts.echo_message -> echo_message
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
        input_model = self._load_model(
            spec.input_model, module_paths=DEFAULT_TOOL_MODEL_MODULE_PATHS
        )
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

    # ------------------------------ Internal API ------------------------------

    def _load_and_validate_tools(self) -> Dict[str, ToolSpec]:
        """
        Load tool specs from either a directory (new layout) or a single
        JSON file (legacy). Optionally sanity-check that each tool's
        import_path resolves to a callable.
        """
        path = self.registry_path

        # New: directory-based registry
        if path.is_dir():
            tools = load_tool_registry(
                path
            )  # Dict[str, ToolSpec]; duplicates handled inside
            if self.import_sanity_check:
                self._sanity_check_imports(tools.values())
            logger.info(
                "[ToolRegistry] Loaded %d tools from directory: %s", len(tools), path
            )
            return tools

        # If not
        else:
            raise FileNotFoundError(f"Tool registry directory path not found: {path}")

        # todo: commented out; delete after debugging
        # # Legacy: single JSON file mapping {name: spec}
        # if path.is_file():
        #     tools = self._load_legacy_file(path)
        #     if self.import_sanity_check:
        #         self._sanity_check_imports(tools.values())
        #     logger.info(
        #         "[ToolRegistry] Loaded %d tools from legacy file: %s", len(tools), path
        #     )
        #     return tools

    def _sanity_check_imports(self, specs: Iterable[ToolSpec]) -> None:
        """
        Import-time verification that each ToolSpec.import_path points to a callable.
        Helpful to fail fast during startup.
        """
        for tool in specs:
            try:
                module_path, func_name = tool.import_path.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                obj = getattr(mod, func_name, None)
                if obj is None:
                    raise ImportError(f"{func_name} not in {module_path}")
                if not callable(obj):
                    raise TypeError(
                        f"{tool.import_path} resolved to non-callable {type(obj).__name__}"
                    )
            except Exception as e:
                raise ImportError(
                    f"[ToolRegistry] Import check failed for '{tool.name}': {e}"
                ) from e

    def _load_model(
        self,
        dotted: str | type[BaseModel] | None,
        *,
        module_paths: list[str] | None = None,
    ) -> type[BaseModel] | None:
        """
        Resolve and import a Pydantic model class.

        Parameters
        ----------
        dotted : str | type[BaseModel] | None
            - Required first argument.
            - May be given positionally or as a keyword.
            - Accepted values:
                * Fully qualified dotted path string, e.g.
                  "bot0_config_agent.tools.configs.tool_models.SelectFilesInput"
                * Bare class name string, e.g. "SelectFilesInput"
                  (resolved against one or more `module_paths`)
                * A class object itself (subclass of BaseModel)
                * None (returns None)

        module_paths : list[str], optional, keyword-only
            One or more module paths to search when `dotted` is a bare class name.
            Example: ["bot0_config_agent.tools.configs.tool_models"]
            Must be passed by keyword:
                self._load_model("SelectFilesInput", module_paths=[...])

        Returns
        -------
        type[BaseModel] | None
            The resolved Pydantic model class, or None if resolution fails.

        Notes
        -----
        - The `*` in the signature makes `module_paths` keyword-only.
        - This design allows ergonomic usage for the common case:
              self._load_model("MyModel")
          while still requiring explicit naming for less common configuration:
              self._load_model("MyModel", module_paths=["custom.module"])

        Examples
        --------
        # 1) Using a fully qualified path
        >>> spec.input_model = "bot0_config_agent.tools.configs.tool_models.ReadFilesInput"
        >>> input_model = self._load_model(spec.input_model)
        >>> type(input_model)
        <class 'bot0_config_agent.tools.configs.tool_models.ReadFilesInput'>

        # 2) Using a bare class name with default module paths
        >>> spec.input_model = "ReadFilesInput"
        >>> input_model = self._load_model(spec.input_model)
        >>> type(input_model)
        <class 'bot0_config_agent.tools.configs.tool_models.ReadFilesInput'>

        # 3) Using a bare class name with a custom module path
        >>> spec.input_model = "CustomInput"
        >>> input_model = self._load_model(
        ...     spec.input_model,
        ...     module_paths=["bot0_config_agent.tools.validators"]
        ... )
        >>> type(input_model)
        <class 'bot0_config_agent.tools.validators.CustomInput'>
        """
        if dotted is None:
            return None
        if isinstance(dotted, type) and issubclass(dotted, BaseModel):
            return dotted
        if not isinstance(dotted, str):
            raise TypeError(f"Expected str or BaseModel subclass, got {type(dotted)}")

        # Default search modules for bare names
        module_paths = module_paths or DEFAULT_TOOL_MODEL_MODULE_PATHS

        try:
            if "." in dotted:
                module_path, cls_name = dotted.rsplit(".", 1)
                module = importlib.import_module(module_path)
            else:
                cls_name = dotted
                last_err = None
                for module_path in module_paths:
                    try:
                        module = importlib.import_module(module_path)
                        model_cls = getattr(module, cls_name)
                        if issubclass(model_cls, BaseModel):
                            return model_cls
                    except Exception as e:
                        last_err = e
                raise ImportError(
                    f"Could not resolve {cls_name} in {module_paths}: {last_err}"
                )

            model_cls = getattr(module, cls_name)
            if not issubclass(model_cls, BaseModel):
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
            >>> fn = self._load_transform_fn("bot0_config_agent.tools.tool_transformation.dir_to_files")
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

    # todo: commented out; delete later
    # def _load_legacy_file(self, path: Path) -> Dict[str, ToolSpec]:
    #     """
    #     Load a legacy single-file registry JSON.

    #     Expected format:
    #         {
    #         "tool_name": {
    #             "name": "...",
    #             "description": "...",
    #             "import_path": "pkg.module.fn",
    #             "parameters": {...},
    #             "input_model": "OptionalModelName",
    #             "output_model": "OptionalModelName"
    #         },
    #         ...
    #         }

    #     Args:
    #         path: Path to the registry JSON file.

    #     Returns:
    #         Dict[str, ToolSpec]: Mapping of tool name → validated ToolSpec.

    #     Raises:
    #         FileNotFoundError: If the file does not exist.
    #         ValueError: If the JSON structure is invalid or not a dict.
    #         ImportError: If import_path does not resolve to a callable.
    #     """
    #     if not path.exists():
    #         raise FileNotFoundError(f"Legacy registry file not found: {path}")

    #     try:
    #         raw = json.loads(path.read_text(encoding="utf-8"))
    #     except Exception as e:
    #         raise RuntimeError(f"Failed reading legacy registry file {path}") from e

    #     if not isinstance(raw, dict):
    #         raise ValueError(
    #             f"Legacy registry must be a JSON object mapping tool names to specs; "
    #             f"got {type(raw).__name__}"
    #         )

    #     validated: Dict[str, ToolSpec] = {}
    #     for name, spec in raw.items():
    #         tool = ToolSpec.model_validate(spec)

    #         # import-time check for safety
    #         try:
    #             module_path, func_name = tool.import_path.rsplit(".", 1)
    #             mod = importlib.import_module(module_path)
    #             if not hasattr(mod, func_name):
    #                 raise ImportError(f"{func_name} not found in {module_path}")
    #         except Exception as e:
    #             raise ImportError(
    #                 f"Failed to import '{tool.import_path}' for '{name}': {e}"
    #             )

    #         validated[name] = tool

    #     return validated
