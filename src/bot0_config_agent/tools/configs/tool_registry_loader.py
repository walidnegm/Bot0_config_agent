"""
tools/configs/tool_registry_loader.py

Utility functions for loading a *tool registry* from a directory.

Policy (clean version)
----------------------
- The directory that contains `_manifest.yaml` is the **registry root**.
- We scan that directory **recursively** for `*.json` files only.
- We do not branch outside the root (symlinks that escape are ignored).
- `_manifest.yaml` is optional and may only contain per-file `overrides`.
- Legacy `include` behavior is removed.

Manifest format
---------------
_manifest.yaml (optional):

  overrides:
    "subdir/tool_spec.json":
      description: "Overridden description"
      parameters:
        properties:
          limit:
            default: 100

Notes
-----
* Duplicate tool names across files are rejected (explicit error).
* Overrides are applied *after* the JSON file is loaded (shallow dict update).
* Files that do not look like a single ToolSpec (e.g., aggregate mappings) are rejected
  with a clear error telling you to split/relocate them.

Public API
----------
- load_tool_registry(dir_: Path = TOOLS_REGISTRY_DIR) -> Dict[str, ToolSpec]
"""

from __future__ import annotations
import logging
from pathlib import Path
import json
from typing import Dict, Any, Iterator, Sequence
import yaml

from bot0_config_agent.tools.configs.tool_models import ToolSpec  # Pydantic model
from bot0_config_agent.configs.paths import TOOLS_REGISTRY_DIR

logger = logging.getLogger(__name__)


# Scan all JSONs since you said you don't use *.tool.json suffixes.
JSON_GLOB = "**/*.json"

# Minimal shape check before Pydantic validation
_REQUIRED_TOP_LEVEL_FIELDS: Sequence[str] = (
    "name",
    "description",
    "import_path",
    "parameters",
)


def _to_posix_relpath(root: Path, p: Path) -> str:
    """Return path of `p` relative to `root` in POSIX form (with `/`)."""
    return str(p.relative_to(root)).replace("\\", "/")


def load_manifest(dir_: Path = TOOLS_REGISTRY_DIR) -> dict:
    """
    Load `_manifest.yaml` from `dir_` if present; otherwise return sane defaults.

    Returns
    -------
    dict
        {
          "overrides": dict[str, dict]  # per-file shallow patches keyed by rel path
        }
    """
    dir_ = dir_.resolve()
    mf = dir_ / "_manifest.yaml"
    if not mf.exists():
        return {"overrides": {}}

    try:
        data = yaml.safe_load(mf.read_text()) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse manifest: {mf}") from e

    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(
            f"manifest.overrides must be a dict, got: {type(overrides).__name__}"
        )

    # Legacy `include` is intentionally ignored in this clean version.
    return {"overrides": overrides}


def iter_tool_files(root: Path) -> Iterator[Path]:
    """
    Yield candidate tool spec files under `root`, recursively.

    Returns an **iterator** of Paths (generator). We purposely type the return as
    Iterator[Path] to signal single-pass consumption.

    Only walks within `root` (no branching to adjacent dirs); symlinks that
    escape the root are ignored.
    """
    root_resolved = root.resolve()
    for f in root.rglob(JSON_GLOB):
        # Skip the manifest and non-files early
        if f.name == "_manifest.yaml":
            continue

        # Guard against symlinks that escape root
        try:
            f_resolved = f.resolve()
            f_resolved.relative_to(root_resolved)
        except Exception:
            continue

        if f_resolved.is_file():
            yield f_resolved


def load_tool_registry(dir_: Path = TOOLS_REGISTRY_DIR) -> Dict[str, ToolSpec]:
    """
    Load all tool specs under `dir_` (registry root), validate them,
    apply per-file overrides, and return a mapping of tool name -> ToolSpec.

    Parameters
    ----------
    dir_ : Path
        Registry root directory containing one or more `*.json` files and
        an optional `_manifest.yaml`.

    Returns
    -------
    Dict[str, ToolSpec]
        Mapping from `ToolSpec.name` to validated ToolSpec instances.

    Raises
    ------
    ValueError
        - If duplicate tool names are encountered.
        - If a file contains invalid JSON or fails validation.
        - If a file looks like an aggregate/multi-tool mapping or is missing required fields.
    RuntimeError
        - If manifest parsing fails or files cannot be read.
    """
    dir_ = dir_.resolve()

    # Minimum logging
    files = list(iter_tool_files(dir_))
    logger.info(
        "[loader] registry=%s | module=%s | %d JSON file(s) found",
        dir_,
        __file__,
        len(files),
    )

    mf = load_manifest(dir_)
    tools: Dict[str, ToolSpec] = {}

    for f in iter_tool_files(dir_):
        spec_dict: Dict[str, Any] = _load_json_file(f)
        rel_key = _to_posix_relpath(dir_, f)

        if not isinstance(spec_dict, dict):
            raise ValueError(f"File does not contain an object at top-level: {rel_key}")

        if not _looks_like_single_toolspec(spec_dict):
            # Helpful, explicit error for the common mistake (aggregate dicts)
            raise ValueError(
                f"File '{rel_key}' does not look like a single ToolSpec "
                f"({_missing_required_fields_msg(spec_dict)}). "
                "If this is an aggregate mapping (multiple tools in one file), split them "
                "into separate files or move this file out of the registry root."
            )

        # Apply per-file shallow overrides by relative POSIX path
        overrides: Dict[str, Any] = mf.get("overrides", {}).get(rel_key, {})
        if overrides:
            spec_dict.update(overrides)

        # Validate into a ToolSpec
        try:
            tool = ToolSpec(**spec_dict)
        except Exception as e:
            raise ValueError(f"Validation failed for {rel_key}: {e}") from e

        # Check duplicates
        if tool.name in tools:
            existing_file = tools[tool.name].__dict__.get("__source_file__")
            msg = f"Duplicate tool name '{tool.name}' from {rel_key}"
            if existing_file:
                msg += f" (already defined in {existing_file})"
            raise ValueError(msg)

        # Preserve source filename for diagnostics (non-model field).
        tool.__dict__["__source_file__"] = rel_key
        tools[tool.name] = tool

    return tools


def _load_json_file(p: Path) -> Dict[str, Any]:
    """Strict JSON loader with friendlier error messages."""
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        loc = f"{p}:{e.lineno}:{e.colno}"
        raise ValueError(f"Invalid JSON in {loc} — {e.msg}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read {p}") from e


def _looks_like_single_toolspec(d: Dict[str, Any]) -> bool:
    """Heuristic: a single ToolSpec must have the required top-level fields."""
    return all(k in d for k in _REQUIRED_TOP_LEVEL_FIELDS)


def _missing_required_fields_msg(d: Dict[str, Any]) -> str:
    missing = [k for k in _REQUIRED_TOP_LEVEL_FIELDS if k not in d]
    return f"missing required field(s): {', '.join(missing)}"
