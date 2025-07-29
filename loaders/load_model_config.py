"""loader/load_model_config.py"""

from pathlib import Path
import yaml
import logging
from typing import Optional
from loaders.model_configs_models import LoaderConfigEntry
from utils.find_root_dir import find_project_root
from config.paths import MODEL_CONFIG_YAML_FILE

logger = logging.getLogger(__name__)


def load_model_config(
    model_name: str, config_file: Path = MODEL_CONFIG_YAML_FILE
) -> LoaderConfigEntry:
    """
    Load model configuration from models.yaml.

    Args:
        model_name (str): Model key in models.yaml.
        model_configs_yaml_file: Defaults to models.yaml in root/config.

    Returns:
        LoaderConfigEntry: Model config pydantic model with loader and parameters.
    """
    config_path = MODEL_CONFIG_YAML_FILE
    try:
        with open(config_file, "r") as f:
            all_configs = yaml.safe_load(f)

        if model_name not in all_configs:
            raise ValueError(f"Model name '{model_name}' not found in {config_path}")

        entry = all_configs[model_name]
        logger.info(f"✅ Model config for '{model_name}' loaded from {config_path}")

        return LoaderConfigEntry.parse_with_loader(
            model_name, entry
        )  # parses loader + config keys

    except Exception as e:
        raise FileNotFoundError(f"Failed to load config from {config_path}: {e}")
