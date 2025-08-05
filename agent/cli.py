"""
cli.py
🧠 Natural Language CLI for Config Manager

Usage examples:

# Run interactively with a local model
python agent/cli.py --local-model llama_2_7b_chat

# Run interactively with a cloud API model (e.g., OpenAI, Anthropic, Gemini)
python agent/cli.py --api-model gpt-4o

# Run one-off command with a local model
python agent/cli.py --local-model llama_2_7b_chat --once "where are my model files"

# Run one-off command with a cloud API model
python agent/cli.py --api-model claude-3-haiku-20240307 --once "summarize project config"
python -m agent.cli --api-model gpt-4.1-mini --once "where are my config files?"
python -m agent.cli --api-model claude-sonnet-4-20250514 --once "First find all config files in the project (excluding venv, models, etc.), then summarize each."


# Show all available models and their descriptions
python agent/cli.py --show-models-help

Notes:
- You must specify exactly one of --local-model or --api-model.
- Use --show-models-help to see all available model options.
"""

import sys
import os
import logging
import pprint
import argparse
import datetime
from pathlib import Path
from typing import List, Tuple, Any, Dict
from tabulate import tabulate
from agent.core import AgentCore
from utils.get_model_info_utils import (
    get_local_model_names,
    get_api_model_names,
    get_local_models_and_help,
    get_api_models_and_help,
    print_all_model_choices,
)
from configs.paths import MODEL_CONFIGS_YAML_FILE
import logging_config

# Setup logger
logger = logging.getLogger(__name__)


def format_file_metadata(file_path: Path | str) -> List[str]:
    """
    Return basic metadata for a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        list[str]: [file path, size as string (KB or MB), created timestamp as string].
                   If any error occurs, size and timestamp are returned as '?'.
    """
    file_path = str(file_path)
    try:
        size = os.path.getsize(file_path)
        created_ts = os.path.getctime(file_path)
        size_str = (
            f"{size / 1024:.1f} KB" if size < 1e6 else f"{size / (1024 * 1024):.1f} MB"
        )
        created_str = datetime.datetime.fromtimestamp(created_ts).strftime(
            "%Y-%m-%d %H:%M"
        )
        return [file_path, size_str, created_str]
    except Exception:
        logger.error(f"Error getting metadata for {file_path}", exc_info=True)
        return [file_path, "?", "?"]


USE_COLOR = True  # Set to False if piping output or in unsupported terminals


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if USE_COLOR else text


def display_result(result: Dict[str, Any]) -> None:
    """Display and log results (for debugging)"""
    tool = result.get("tool", "Unknown Tool")
    status = result.get("status", "ok")

    # Skip noisy tools
    if (
        tool in {"read_file", "aggregate_file_content", "llm_response_async"}
        and status == "ok"
    ):
        return

    message = result.get("message", "")
    print(f"\n🔧 Tool: {tool}")
    print(f"🗨️  Message: {message}")

    # Logging to debug
    logger.info(f"\n🔧 Tool: {tool}")
    logger.info(f"🗨️  Message: {message}")

    result_payload = result.get("result")
    pp = pprint.PrettyPrinter(indent=2, width=100, compact=False)
    if isinstance(result_payload, dict):
        for field in ("matches", "files", "results"):
            items = result_payload.get(field)
            if isinstance(items, list) and items:
                rows = [format_file_metadata(path) for path in items]
                header = f"\n📁 {field.capitalize()}:"
                table = tabulate(
                    rows, headers=["Path", "Size", "Created"], tablefmt="fancy_grid"
                )
                print(f"\n📁 {bold(field.capitalize())}:")
                print(table)

                # Logging for debugging
                logger.info(header)
                logger.info("\n" + table)

        for k, v in result_payload.items():
            if k in {"matches", "files", "results"}:
                continue
            field_str = f"📌 {k}:"
            pretty = pp.pformat(v)

            print(pretty)
            print(field_str)

            # Logging for debugging
            logger.info(field_str)
            logger.info(pretty)
    elif isinstance(result_payload, list):
        print(f"\n📌 Result (list):")
        pp.pprint(result_payload)

        # Logging for debugging
        pretty = pp.pformat(result_payload)
        logger.info("📌 Result payload:")
        logger.info(pretty)
    else:
        print(f"\n📌 Result:")
        print(result_payload)

        # Logging for debugging
        pretty = pp.pformat(result_payload)
        logger.info("📌 Result payload:")
        logger.info(pretty)


def run_agent_loop(agent: AgentCore) -> None:
    print("🧠 Config Manager CLI (type 'quit' or Ctrl+C to exit)")
    while True:
        try:
            instruction = input("\n📝 Instruction: ").strip()
            if instruction.lower() in {"quit", "exit"}:
                print("👋 Goodbye!")
                break
            results = agent.handle_instruction(instruction)
            print("\n--- Results ---")
            for result in results:
                display_result(result)
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            logger.error(f"Error in agent loop: {e}", exc_info=True)
            print(f"❌ Error: {e}")


def main():
    # Prepare model lists and help info
    try:
        local_model_names = get_local_model_names(MODEL_CONFIGS_YAML_FILE)
        api_model_names = get_api_model_names()
        local_models = get_local_models_and_help(MODEL_CONFIGS_YAML_FILE)
        api_models = get_api_models_and_help()
    except Exception as e:
        logger.error("Failed to load model information", exc_info=True)
        print("❌ Failed to load model info. Exiting.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Agent-powered CLI for natural language instructions"
    )
    parser.add_argument("--once", type=str, help="Run once with a single instruction")
    parser.add_argument(
        "--local-model",
        type=str,
        choices=local_model_names,
        help="Select a local LLM model",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        choices=api_model_names,
        help="Select a cloud API LLM model",
    )
    parser.add_argument(
        "--show-models-help",
        action="store_true",
        help="Show all models with help and exit",
    )

    args = parser.parse_args()

    # Print help and exit if requested
    if args.show_models_help:
        print_all_model_choices(local_models, api_models)
        sys.exit(0)

    # Safe guards:
    chosen_models = [x for x in [args.local_model, args.api_model] if x]
    if len(chosen_models) > 1:
        parser.error(
            "Please specify only one model (local or cloud), not multiple at once."
        )
    if not chosen_models:
        parser.error(
            "Please specify one model: --local-model (local) or --api-model (cloud)."
        )

    # Instantiate agent with the right backend
    try:
        agent = AgentCore(
            local_model_name=args.local_model, api_model_name=args.api_model
        )
    except Exception as e:
        logger.error(f"Failed to initialize AgentCore: {e}", exc_info=True)
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)

    # One-off or interactive mode
    if args.once:
        instruction = args.once.strip()
        try:
            results = agent.handle_instruction(instruction)
            for result in results:
                display_result(result)
        except Exception as e:
            logger.error(f"Error during instruction execution: {e}", exc_info=True)
            print(f"❌ Error: {e}")
    else:
        run_agent_loop(agent)


if __name__ == "__main__":
    main()
