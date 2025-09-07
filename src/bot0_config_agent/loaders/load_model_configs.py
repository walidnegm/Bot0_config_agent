"""loader/load_model_config.py

Load model configs with optional per-user overrides.

Precedence (later wins):
  1) Base file: MODEL_CONFIGS_YAML_FILE (tracked)
  2) Optional override: LOCAL_OVERRIDE_MODEL_CONFIGS_YAML_FILE (gitignored)

Both YAMLs should be dicts keyed by model names, e.g.:

mistral-7b-awq:
  loader: awq
  params:
    max_new_tokens: 2048
    fuse_layers: false
"""

import os
from pathlib import Path
import logging
from typing import cast, Optional, Dict, Any
import yaml

from bot0_config_agent.loaders.model_configs_models import LoaderConfigEntry
from bot0_config_agent.configs.paths import (
    ROOT_DIR,
    MODELS_DIR,
    OFFLOAD_DIR,
    MODEL_CONFIGS_YAML_FILE,
    LOCAL_OVERRIDE_MODEL_CONFIGS_YAML_FILE,
)

logger = logging.getLogger(__name__)

# ----- token substitution + env + absolutize -----
TOKEN_MAP = {
    "<PROJECT_ROOT>": str(ROOT_DIR),
    "<MODELS_DIR>": str(MODELS_DIR),
    "<OFFLOAD_DIR>": str(OFFLOAD_DIR),
}


def deep_merge(
    base: Dict[str, Any] | None, override: Dict[str, Any] | None
) -> Dict[str, Any]:
    """
    Recursively merge `override` into `base`.

    - If a key maps to dicts in both, merge recursively.
    - Otherwise, the override value replaces the base value.
    - Returns a new dict; does not mutate the input objects.

    Args:
        base: The original dictionary (may be None).
        override: The dictionary to overlay (may be None).

    Returns:
        A new dictionary with overrides applied.
    """
    if not base:
        base = {}
    if not override:
        return dict(base)

    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_model_configs(
    model_name: str,
    configs_file: Path = MODEL_CONFIGS_YAML_FILE,
    configs_override_file: Optional[Path] = LOCAL_OVERRIDE_MODEL_CONFIGS_YAML_FILE,
) -> LoaderConfigEntry:
    """
    Load a model config by name from a base YAML and layer an optional local override.

    Precedence:
        base (configs_file) → override (configs_override_file)

    The YAMLs are expected to be dicts keyed by model name. The selected entry is
    validated and normalized into a LoaderConfigEntry via `parse_with_loader`.

    Args:
        model_name: The key of the model config to load (e.g., "mistral-7b-awq").
        configs_file: Path to the base, tracked YAML (default: MODEL_CONFIGS_YAML_FILE).
        configs_override_file: Optional path to a gitignored local override YAML
            (default: LOCAL_OVERRIDE_MODEL_CONFIGS_YAML_FILE). If present, its values
            will deep-merge over the base.

    Returns:
        LoaderConfigEntry: Validated config (loader + params) for the requested model.

    Raises:
        FileNotFoundError: When the base file cannot be read or parsed.
        ValueError: If the requested model name is not found in the merged config.
    """

    # Load base
    with open(configs_file, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    if not isinstance(base_cfg, dict):
        raise ValueError(f"Base config at {configs_file} must be a mapping.")

    merged_cfg = dict(base_cfg)

    # Optional override
    if configs_override_file and Path(configs_override_file).exists():
        with open(configs_override_file, "r", encoding="utf-8") as f:
            override_cfg = yaml.safe_load(f) or {}
        if not isinstance(override_cfg, dict):
            raise ValueError(
                f"Override config at {configs_override_file} must be a mapping."
            )
        merged_cfg = deep_merge(merged_cfg, override_cfg)
        logger.info("🔄 Applied overrides from %s", configs_override_file)
    else:
        logger.debug("No override file found at %s", configs_override_file)

    # Select model entry
    if model_name not in merged_cfg:
        raise ValueError(f"Model name '{model_name}' not found in {configs_file}")

    entry = merged_cfg[model_name]

    entry = _subst_tokens(entry, TOKEN_MAP)
    entry = _absolutize_paths(entry)
    entry = _ensure_defaults(entry)

    logger.info(
        "✅ Model config for '%s' loaded (base: %s, override: %s)",
        model_name,
        configs_file,
        configs_override_file,
    )

    # ----- Validate/normalize with your existing schema -----
    try:
        return LoaderConfigEntry.parse_with_loader(model_name, entry)
    except Exception as e:
        # Don’t mislabel schema/validation as file-not-found; bubble clearly.
        raise ValueError(f"Invalid model config for '{model_name}': {e}") from e


def _absolutize_paths(x):
    """Recursively absolutize any string that looks pathlike and isn’t absolute."""
    if isinstance(x, str) and _is_pathlike(x):
        p = Path(x)
        if not p.is_absolute():
            return str((ROOT_DIR / p).resolve())
        return str(p)
    if isinstance(x, list):
        return [_absolutize_paths(i) for i in x]
    if isinstance(x, dict):
        return {k: _absolutize_paths(v) for k, v in x.items()}
    return x


def _ensure_defaults(entry: dict) -> dict:
    """Ensure reasonable defaults like offload_folder and local_files_only."""
    load = entry.setdefault("load_config", {})

    # Default offload folder if missing and we’re not using llama.cpp natively
    if "offload_folder" not in load and entry.get("loader") not in {"llama_cpp"}:
        load["offload_folder"] = str(OFFLOAD_DIR)

    # If model_id_or_path is a real local path, prefer local_files_only=True unless already set
    mid = load.get("model_id_or_path")
    if isinstance(mid, str):
        p = Path(mid)
        if p.exists() and "local_files_only" not in load:
            load["local_files_only"] = True
    return entry


def _is_pathlike(s: str) -> bool:
    return isinstance(s, str) and ("/" in s or "\\" in s)


def _subst_tokens(x, token_map):
    """Recursively replace <TOKENS> and ${ENV_VARS} in strings."""
    if isinstance(x, str):
        for k, v in token_map.items():
            if k in x:
                x = x.replace(k, v)
        # expand ${ENV_VAR}
        x = os.path.expandvars(x)
        return x
    if isinstance(x, list):
        return [_subst_tokens(i, token_map) for i in x]
    if isinstance(x, dict):
        return {k: _subst_tokens(v, token_map) for k, v in x.items()}
    return x
