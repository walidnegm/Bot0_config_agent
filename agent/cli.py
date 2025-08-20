"""
agent/cli.py
Command-line interface for the agent.
"""
import sys
import os
import argparse
import logging
import json

# Determine the project root directory (the parent of the 'agent' directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.core import AgentCore
from tools.tool_models import ToolResult
from utils.get_llm_api_keys import get_openai_api_key, get_anthropic_api_key, get_google_api_key
from configs.api_models import get_llm_provider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    try:
        agent = AgentCore(local_model_name=args.local_model, api_model_name=args.api_model)
    except Exception as e:
        logger.error(f"Failed to initialize AgentCore: {e}", exc_info=True)
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
        # Interactive mode can be implemented here
        print("Interactive mode is not yet implemented. Use the --once flag.")

if __name__ == "__main__":
    main()
