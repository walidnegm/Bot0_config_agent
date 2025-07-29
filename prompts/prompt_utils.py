"""
prompts/utils/prompt_utils.py

Helper functions to for loading/generating/validating prompts.
"""

from pathlib import Path
import logging
from typing import Any, Dict, Optional, Tuple
import yaml
from config.paths import PROMPTS_CONFIG, TOOL_AGENT_PROMPTS

logger = logging.getLogger(__name__)


def load_prompt_config() -> Dict[str, Any]:
    """
    Loads and validates the prompt configuration from `prompts_config.yaml`.

    Resolves the relative path to the prompt template file and adds it to
    the returned dictionary under
    the key `prompt_templates.resolved_template_path`.

    Expected YAML schema:
        prompt_templates:
        template_file: "../prompts/chat_templates.yaml"
        active_templates:
            system: "default"
            user: "question"

    Returns:
        dict: The full config dictionary with an additional absolute key:
              config["prompt_templates"]["resolved_template_path"]

    Raises:
        FileNotFoundError: If the config file or template file doesn't exist.
        KeyError: If expected keys are missing in the YAML structure.
        yaml.YAMLError: If the YAML is malformed.
    """
    logger.info(f"[load_prompt_config] 📄 Loading from: {PROMPTS_CONFIG}")

    try:
        with open(PROMPTS_CONFIG, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"[load_prompt_config] ❌ Not found: {PROMPTS_CONFIG}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"[load_prompt_config] ❌ YAML error: {e}")
        raise

    try:
        prompt_templates = config["prompt_templates"]
    except KeyError as e:
        raise KeyError(f"Missing key in config file: {e}")

    resolved_path = TOOL_AGENT_PROMPTS.resolve()
    prompt_templates["resolved_template_path"] = str(resolved_path)
    logger.info(f"[load_prompt_config] ✅ Resolved template: {resolved_path}")

    return config


def load_prompt_templates(template_path: Path) -> Dict[str, str]:
    """
    Load the prompt templates from the given YAML file.

    Args:
        template_path (Path): Path to YAML template file.

    Returns:
        Dict[str, str]: Mapping of prompt keys to strings.

    Raises:
        FileNotFoundError, ValueError, yaml.YAMLError
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {template_path}")

    logger.info(f"[load_prompt_templates] 📄 Reading: {template_path}")

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            templates = yaml.safe_load(f)
        if not isinstance(templates, dict):
            raise ValueError("Prompt template file must contain a dict of templates.")
        return templates
    except yaml.YAMLError as e:
        logger.error(f"[load_prompt_templates] ❌ YAML error: {e}")
        raise


def get_prompts() -> Tuple[str, str]:
    """
    Resolve and return the active (system, user) prompts.

    Returns:
        Tuple[str, str]: (system_prompt, user_prompt)

    Raises:
        KeyError: If active keys are missing or not in template file.
    """
    config = load_prompt_config()
    resolved_path = Path(config["prompt_templates"]["resolved_template_path"])
    templates = load_prompt_templates(resolved_path)

    active = config["prompt_templates"]["active_templates"]
    system_key = active.get("system")
    user_key = active.get("user")

    try:
        return templates[system_key], templates[user_key]
    except KeyError as e:
        raise KeyError(f"Missing template key in file: {e}")
