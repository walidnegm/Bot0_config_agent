"""
agent/cli.py
Command-line interface for the agent.
"""
import sys
import os
from pathlib import Path
import argparse
import logging
import json
import yaml
from typing import List
from huggingface_hub import list_repo_files, snapshot_download, get_hf_file_metadata

# Determine the project root directory (the parent of the 'agent' directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.core import AgentCore
from tools.tool_models import ToolResult
from utils.get_llm_api_keys import get_openai_api_key, get_anthropic_api_key, get_google_api_key
from configs.api_models import get_llm_provider
from configs.paths import MODEL_CONFIGS_YAML_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_available_models(config_file: Path = MODEL_CONFIGS_YAML_FILE) -> List[str]:
    """Load available local model names from model_configs.yaml."""
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        return list(config_data.keys())
    except Exception as e:
        logger.error(f"Failed to load model configurations from {config_file}: {e}")
        return []

def display_result(result_data: dict):
    """Prints a formatted tool result to the console."""
    print(f"\n🔧 Tool: {result_data.get('tool')}")
    print(f"🗨️  Message: {result_data.get('message')}")
    print("📌 Result:")
    # Pretty print the result if it's a dict or list
    result_payload = result_data.get('result')
    if isinstance(result_payload, (dict, list)):
        print(json.dumps(result_payload, indent=2, ensure_ascii=False))
    else:
        print(result_payload)

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description="A CLI for interacting with an AI agent.")
    parser.add_argument("--local-model", help="Specify the local model to use.")
    parser.add_argument("--api-model", help="Specify the API model to use.")
    parser.add_argument("--once", help="Run a single instruction and exit.")
    args = parser.parse_args()

    # Validate that exactly one model is chosen
    if (args.local_model and args.api_model) or (not args.local_model and not args.api_model):
        parser.error("Please specify exactly one of --local-model or --api-model.")

    # Validate API key if using API model
    if args.api_model:
        try:
            provider = get_llm_provider(args.api_model)
            if provider == "openai":
                get_openai_api_key()
            elif provider == "anthropic":
                get_anthropic_api_key()
            elif provider == "gemini":
                get_google_api_key()
        except ValueError as e:
            logger.error(f"Failed to validate API key: {e}")
            sys.exit(1)

    # Validate Hugging Face token and model cache for local models
    if args.local_model:
        try:
            with open(MODEL_CONFIGS_YAML_FILE, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            if args.local_model not in config_data:
                available_models = get_available_models()
                logger.error(
                    f"Model '{args.local_model}' not found in {MODEL_CONFIGS_YAML_FILE}. "
                    f"Available models: {', '.join(available_models) if available_models else 'None'}"
                )
                sys.exit(1)
            loader = config_data[args.local_model].get("loader")
            repo_id = config_data[args.local_model].get("config", {}).get("model_id_or_path", "")
            if loader in ["transformers", "gptq", "awq"]:
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    logger.error(
                        "HF_TOKEN environment variable not set. "
                        "Required for gated models like LiquidAI/LFM2-1.2B. "
                        "Set HF_TOKEN or run `huggingface-cli login`."
                    )
                    sys.exit(1)
                cache_dir = os.path.join(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
                model_cache_path = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")
                if not os.path.exists(model_cache_path):
                    logger.info(f"Model {repo_id} not found in cache: {model_cache_path}. Attempting to download...")
                    try:
                        snapshot_download(repo_id=repo_id, token=hf_token)
                    except Exception as e:
                        logger.error(f"Failed to download model {repo_id}: {e}")
                        if "401 Client Error" in str(e):
                            logger.error(
                                f"Authentication failed for {repo_id}. "
                                "Ensure you have access to the model and a valid HF_TOKEN."
                            )
                        sys.exit(1)
                # Find the latest snapshot directory
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if not os.path.exists(snapshots_dir):
                    logger.error(f"No snapshots found for model {repo_id} at {snapshots_dir}")
                    sys.exit(1)
                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if not snapshots:
                    logger.error(f"No snapshot directories found for model {repo_id} at {snapshots_dir}")
                    sys.exit(1)
                # Use the latest snapshot (most recent commit ID)
                latest_snapshot = max(snapshots, key=lambda d: os.path.getmtime(os.path.join(snapshots_dir, d)))
                snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
                required_files = ['config.json', 'tokenizer.json']
                if not all(os.path.exists(os.path.join(snapshot_path, f)) for f in required_files):
                    logger.error(f"Model {repo_id} snapshot at {snapshot_path} is missing required files: {required_files}")
                    sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to validate model configuration: {e}")
            sys.exit(1)

    try:
        agent = AgentCore(local_model_name=args.local_model, api_model_name=args.api_model)
    except (ValueError, FileNotFoundError) as e:
        if "Local model" in str(e) or "Failed to load config" in str(e):
            available_models = get_available_models()
            logger.error(
                f"{e}\nAvailable local models: {', '.join(available_models) if available_models else 'None'}.\n"
                f"Please check {MODEL_CONFIGS_YAML_FILE} or use an API model."
            )
            sys.exit(1)
        logger.error(f"Failed to initialize AgentCore: {e}")
        sys.exit(1)

    if args.once:
        instruction = args.once.strip()
        try:
            logger.info("=" * 50)
            logger.info(f"User Instruction: {instruction}")
            tool_results = agent.handle_instruction(instruction)
            print("\n--- Results ---")
            for i, result in enumerate(tool_results.results):
                display_result(result.model_dump())
                logger.info(
                    "RESULT_JSON for step %d:\n%s",
                    i,
                    json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
                )
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error during instruction execution: {e}", exc_info=True)
            print(f"❌ Error: {e}")
    else:
        print("Interactive mode is not yet implemented. Use the --once flag.")

if __name__ == "__main__":
    main()
