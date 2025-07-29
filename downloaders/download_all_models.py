import os
from pathlib import Path
import logging
import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.find_root_dir import find_project_root
from config.paths import MODEL_CONFIG_YAML_FILE
import logging_config

# Set up logging
logger = logging.getLogger(__name__)


def load_models_yaml(yaml_path: Path) -> dict:
    """Load the YAML config file for models."""
    if not yaml_path.exists():
        logger.error(f"models.yaml not found at {yaml_path}")
        raise FileNotFoundError(f"models.yaml not found at {yaml_path}")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def ensure_hf_token() -> str:
    """Ensure Hugging Face token is loaded and valid."""
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        logger.error(
            "HUGGING_FACE_TOKEN environment variable not set. Please set your Hugging Face token."
        )
        raise ValueError(
            "HUGGING_FACE_TOKEN environment variable not set. Please set your Hugging Face token."
        )
    return hf_token


def cache_and_load_model(model_id: str, hf_token: str, name: str, quant: str) -> bool:
    """
    Cache a model (download if needed) and attempt to load it.
    Returns True on success, False on failure.
    """
    logger.info(f"🔎 Checking cache for {name} ({model_id}) [{quant}] ...")
    try:
        # Download/check cache
        local_path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            local_files_only=False,  # Set to True to only check local
        )
        logger.info(f"📦 Model {name} ({model_id}) is now cached at {local_path}.")
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model_obj = AutoModelForCausalLM.from_pretrained(local_path)
        logger.info(f"✅ Model loaded from {local_path}.")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to cache or load {name} ({model_id}): {e}")
        return False


def main():
    project_root = find_project_root()
    CONFIG_YAML_FILE = MODEL_CONFIG_YAML_FILE  # use model_configs.yaml instead

    if not CONFIG_YAML_FILE.exists():
        raise FileNotFoundError(f"models.yaml file {CONFIG_YAML_FILE} not found.")

    models_config = load_models_yaml(CONFIG_YAML_FILE)
    hf_token = ensure_hf_token()

    success, failed = [], []

    only_include = [
        # "lfm2_1_2b",
        "tinyllama_1_1b_chat_gguf",
    ]  # Optional to download just selected models

    for name, config in models_config.items():
        if name not in only_include:
            continue

        model_id = config["model_id"]
        quant = config.get("quantization", "unknown")
        ok = cache_and_load_model(model_id, hf_token, name, quant)
        (success if ok else failed).append(name)

    logger.info("🎉 All model cache checks/downloads attempted.")
    logger.info(f"✅ Success: {success}")
    if failed:
        logger.warning(f"❌ Failed: {failed}")


if __name__ == "__main__":
    main()
