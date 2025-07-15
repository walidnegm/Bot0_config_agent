import os
from pathlib import Path
import logging
import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_project_root(
    markers=(".git", "requirements.txt", "pyproject.toml", "README.md")
) -> Path:
    """Search up parent directories for a project root marker file."""
    path = Path(__file__).resolve().parent
    for parent in [path] + list(path.parents):
        if any((parent / marker).exists() for marker in markers):
            logger.info(f"🔍 Project root found: {parent}")
            return parent
    logger.warning("⚠️ Project root not found, using CWD.")
    return Path.cwd()


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
    yaml_path = project_root / "models.yaml"
    models_config = load_models_yaml(yaml_path)
    hf_token = ensure_hf_token()

    success, failed = [], []
    for model in models_config["models"]:
        model_id = model["id"]
        name = model["name"]
        quant = model.get("quantization", "unknown")
        ok = cache_and_load_model(model_id, hf_token, name, quant)
        (success if ok else failed).append(name)

    logger.info("🎉 All model cache checks/downloads attempted.")
    logger.info(f"✅ Success: {success}")
    if failed:
        logger.warning(f"❌ Failed: {failed}")


if __name__ == "__main__":
    main()
