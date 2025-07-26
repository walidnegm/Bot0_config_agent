"""loader/load_model_config.py"""

from pathlib import Path
import json
import logging
from typing import Optional
from loaders.model_config_models import LoaderConfigEntry
from utils.find_root_dir import find_project_root

logger = logging.getLogger(__name__)

MODEL_CONFIGS_JSON = "model_configs.json"


def load_model_config(
    model_name: str, json_path: Optional[Path] = None
) -> LoaderConfigEntry:
    """
    Load model configuration from models_configs.json.

    Args:
        model_name (str): Model key in models_configs.json.
        json_path (Optional[Path]): Path to the JSON config file.
            Defaults to models_configs.json in project root.

    Returns:
        ModelConfigEntry: Model config pydantic model with loader and parameters.
    """
    config_path = json_path or find_project_root() / "loaders" / MODEL_CONFIGS_JSON
    try:
        with open(config_path, "r") as f:
            all_configs = json.load(f)
        if model_name not in all_configs:
            raise ValueError(f"Model name '{model_name}' not found in {config_path}")
        logger.info(f"✅ Model config for '{model_name}' loaded from {config_path}")
        return LoaderConfigEntry(**all_configs[model_name])  # -> pydantic model
    except Exception as e:
        raise FileNotFoundError(f"Failed to load config from {config_path}: {e}")
