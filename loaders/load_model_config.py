"""
loaders/load_model_config.py
Loads model configurations from a YAML file.
"""

import yaml
from pathlib import Path
import logging
from typing import Optional
from pydantic import ValidationError
from loaders.model_configs_models import LoaderConfigEntry
from utils.find_root_dir import find_project_root
from configs.paths import MODEL_CONFIGS_YAML_FILE

logger = logging.getLogger(__name__)

def load_model_config(
    model_name: str, config_file: Path = MODEL_CONFIGS_YAML_FILE
) -> LoaderConfigEntry:
    """
    Load model configuration from models.yaml.

    Args:
        model_name (str): Model key in models.yaml.
        config_file (Path): Path to the YAML configuration file.

    Returns:
        LoaderConfigEntry: Model config Pydantic model with loader and parameters.

    Raises:
        ValueError: If the model is not found, YAML is invalid, or entry is malformed.
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    config_path = config_file
    try:
        with open(config_path, "r") as f:
            all_configs = yaml.safe_load(f)
        if not isinstance(all_configs, dict):
            logger.error(f"Invalid YAML format in {config_path}: Expected a dictionary, got {type(all_configs)}")
            raise ValueError(f"Invalid YAML format in {config_path}: Expected a dictionary")
        if not all_configs:
            logger.error(f"Empty or invalid YAML file: {config_path}")
            raise ValueError(f"Empty or invalid YAML file: {config_path}")
        if model_name not in all_configs:
            logger.error(f"Model name '{model_name}' not found in {config_path}")
            raise ValueError(f"Model name '{model_name}' not found in {config_path}")
        entry = all_configs[model_name]
        if not isinstance(entry, dict):
            logger.error(f"Invalid config entry for '{model_name}' in {config_path}: Expected a dictionary, got {type(entry)}")
            raise ValueError(f"Invalid config entry for '{model_name}': Expected a dictionary")
        required_fields = ['loader', 'model_id', 'quantization', 'config', 'generation_config']
        missing_fields = [field for field in required_fields if field not in entry]
        if missing_fields:
            logger.error(f"Missing required fields for '{model_name}' in {config_path}: {missing_fields}")
            raise ValueError(f"Missing required fields for '{model_name}': {missing_fields}")
        logger.debug(f"Raw config for '{model_name}': {entry}")
        logger.info(f"Processing loader for '{model_name}': {entry['loader']}")
        try:
            config_entry = LoaderConfigEntry.parse_with_loader(model_name, entry)
            logger.debug(f"Parsed LoaderConfigEntry for '{model_name}': loader={config_entry.loader}, config_type={type(config_entry.config).__name__}")
            return config_entry
        except ValidationError as e:
            logger.error(f"Failed to validate LoaderConfigEntry for '{model_name}': {e}")
            raise ValueError(f"Failed to validate LoaderConfigEntry for '{model_name}': {e}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file {config_path}: {e}")
        raise ValueError(f"Failed to parse YAML file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_path}: {e}")
        raise ValueError(f"Unexpected error loading config: {e}")
