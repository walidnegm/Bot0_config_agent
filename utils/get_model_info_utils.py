"""utils/get_model_info_utils.py"""

from pathlib import Path
import logging
from typing import List, Union, Tuple
from yaml import safe_load
from configs.paths import MODEL_CONFIGS_YAML_FILE
from configs.api_models import ANTHROPIC_MODELS, OPENAI_MODELS, GEMINI_MODELS

logger = logging.getLogger(__name__)


# Functions to gather model names and help info
def get_local_model_names(config_path: Path = MODEL_CONFIGS_YAML_FILE) -> List:
    """Get local LLM model names from model_configs.yaml file"""
    try:
        with open(config_path, "r") as f:
            config = safe_load(f)
        return list(config.keys())
    except Exception as e:
        logger.error(f"⚠️ Failed to load model configs: {e}")
        return []


def get_local_models_and_help(
    config_path: Path = MODEL_CONFIGS_YAML_FILE,
) -> List[Tuple[str, str]]:
    """Load local models and their help strings from YAML."""
    try:
        with open(config_path, "r") as f:
            config = safe_load(f)
        models = []
        for name, entry in config.items():
            desc = entry.get("description", f"no description for {name}")
            models.append((name, desc))
        return models
    except Exception as e:
        logger.error(f"⚠️ Failed to load local models: {e}")
        return []


def get_api_model_names() -> List[str]:
    """Return just the model names for argparse choices."""
    return [name for name, _ in get_api_models_and_help()]


def get_api_models_and_help() -> List[Tuple[str, str]]:
    """Return a flat list of (model_name, help_text) for all cloud API models."""
    models: List[Tuple[str, str]] = []
    for name, desc in OPENAI_MODELS.items():
        models.append((name, desc))
    for name, desc in ANTHROPIC_MODELS.items():
        models.append((name, desc))
    for name, desc in GEMINI_MODELS.items():
        models.append((name, desc))
    return models


def print_api_models_help() -> None:
    print("Available cloud models:")
    for name, help_text in get_api_models_and_help():
        print(f"  {name:20}  {help_text}")


def print_all_model_choices(local_models, api_models, use_color=True):
    # Helper for color/bold
    def bold(text):
        return f"\033[1m{text}\033[0m" if use_color else text

    def faint(text):
        return f"\033[2m{text}\033[0m" if use_color else text

    def cyan(text):
        return f"\033[36m{text}\033[0m" if use_color else text

    # Headers
    print(bold("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    print(f"{cyan('🖥️  Local Models Available')}")
    print(bold("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))

    for model, desc in local_models:
        desc_disp = (
            faint("(no description provided)")
            if not desc or "no description" in desc.lower()
            else desc
        )
        print(f"  {model.ljust(25)} {desc_disp}")

    print()
    print(f"{cyan('☁️  Cloud API Models Available')}")
    print(bold("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    for model, desc in api_models:
        print(f"  {model.ljust(25)} {desc}")
    print(bold("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    print()


# Usage (assuming local_models and api_models are list of tuples)
# local_models = [("qwen3_1_7b_instruct_gptq", ""), ...]
# api_models = [("gpt-3.5-turbo", "Legacy model, fast & cheap, ..."), ...]


# def print_all_model_choices(
#     local_models: List[Tuple[str, str]], api_models: List[Tuple[str, str]]
# ) -> None:
#     print("\nLocal models available:")
#     if local_models:
#         for name, desc in local_models:
#             print(f"  {name:20}  {desc}")
#     else:
#         print("  (none found)")
#     print("\nCloud API models available:")
#     if api_models:
#         for name, desc in api_models:
#             print(f"  {name:20}  {desc}")
#     else:
#         print("  (none found)")
#     print(
#         "\nPick a local model with --model <model_name> or a cloud model with --api-model <api_model_name>\n"
#     )
