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

from pathlib import Path
import logging
from typing import Optional, Dict, Any
import yaml

from loaders.model_configs_models import LoaderConfigEntry
from configs.paths import (
    MODEL_CONFIGS_YAML_FILE,
    LOCAL_OVERRIDE_MODEL_CONFIGS_YAML_FILE,
)

logger = logging.getLogger(__name__)


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
    try:
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
        logger.info(
            "✅ Model config for '%s' loaded (base: %s, override: %s)",
            model_name,
            configs_file,
            configs_override_file,
        )

        # Parse & validate (ensures correct loader + params contract)
        return LoaderConfigEntry.parse_with_loader(model_name, entry)

    except Exception as e:
        # Surface the base file path in the error for easier debugging
        raise FileNotFoundError(
            f"Failed to load config from {configs_file}: {e}"
        ) from e
