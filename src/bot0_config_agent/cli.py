"""
cli.py
==========================
🧠 Natural Language CLI for Config Manager

Interactive and batch command-line interface for the agent-based Config Manager.
Supports both local and API (cloud) LLMs. Handles interactive sessions or one-off commands,
and provides clear, shareable logs and user-friendly output.

Usage examples:
----------------
# Run interactively with a local model:
python -m bot0_config_agent.cli --local-model llama_2_7b_chat

# Run interactively with a cloud API model (e.g., OpenAI, Anthropic, Gemini):
python -m bot0_config_agent.cli --api-model gpt-4o

# Run one-off command with a local model:
python -m bot0_config_agent.cli --local-model llama_2_7b_chat --once "where are my model files"

# Run one-off command with a cloud API model:
python -m bot0_config_agent.cli --api-model gpt-4.1-mini --once "where are my config files?"

# More complex one-off commands:
python -m bot0_config_agent.cli --api-model claude-3-haiku-20240307 --once "summarize project config"
python -m bot0_config_agent.cli --api-model claude-sonnet-4-20250514 --once "First find all config files in the project (excluding venv, models, etc.), then summarize each."
python -m bot0_config_agent.cli --api-model gpt-4.1-mini --once "list all files in the ./bot0_config_agent/agent directory and read the first 3 files."
python -m bot0_config_agent.cli --local-model deepseek_coder_1_3b_gptq --once "list all files in the ./bot0_config_agent/agent directory and read the first file."
python -m bot0_config_agent.cli --local-model lfm2_1_2b --once "list all files in the ./bot0_config_agent/agent directory and summarize the first 3 files."
python -m bot0_config_agent.cli --local-model phi_3_5_mini_awq --once "list all files in the ./bot0_config_agent/agent directory and summarize them."

# Show all available models and their descriptions:
python -m bot0_config_agent.cli --show-models-help

# More examples with exclusions:
python -m bot0_config_agent.cli --api-model gpt-4.1-mini --once "List all files in the ./bot0_config_agent/agent directory excluding __pycache__, .git, and venv."
python -m bot0_config_agent.cli --api-model gpt-4.1-mini --once "List all files in the ./bot0_config_agent/agent directory excluding __pycache__, .git, and venv, then summarize their contents."

Notes:
------
- You must specify exactly one of --local-model or --api-model.
- Use --show-models-help to see all available model options.
"""

import sys
import logging
import pprint
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Any, Dict, Mapping, Sequence, Union
from tabulate import tabulate

# Import from project modules
from bot0_config_agent.agent.core import AgentCore
from bot0_config_agent.agent_models.step_status import StepStatus
from bot0_config_agent.utils.model.get_model_info_utils import (
    get_local_model_names,
    get_api_model_names,
    get_local_models_and_help,
    get_api_models_and_help,
    print_all_model_choices,
)
from bot0_config_agent.configs.paths import MODEL_CONFIGS_YAML_FILE
import bot0_config_agent.logging_config as logging_config  # * Need this configure logging file (Only need to import once at entry point)

# Set up logging
logger = logging.getLogger(__name__)

USE_COLOR = True  # Set to False if piping output or in unsupported terminals


def format_file_metadata(item: Union[str, Path, Mapping[str, Any]]) -> dict:
    """
    Accepts either a path (str/Path) or a dict with keys {"file", "content"}.
    Returns a stable metadata dict without throwing on missing files.
    """
    if isinstance(item, Mapping):
        file_path = item.get("file")
        content = item.get("content", None)
    else:
        file_path = str(item)
        content = None

    if not file_path or not isinstance(file_path, (str, Path)):
        raise ValueError(
            "format_file_metadata: expected path or {'file','content'} dict"
        )

    p = Path(file_path)
    exists = p.exists()

    # Prefer on-disk size; fallback to content length
    try:
        size_bytes = (
            p.stat().st_size
            if exists
            else (len(content.encode("utf-8")) if isinstance(content, str) else None)
        )
    except Exception:
        size_bytes = None

    # Modified time if available
    modified = None
    try:
        if exists:
            ts = p.stat().st_mtime
            modified = datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        modified = None

    # Optional text stats when we have in-memory content
    preview = None
    line_count = None
    if isinstance(content, str):
        lines = content.splitlines()
        line_count = len(lines)
        preview = "\n".join(lines[:30])  # first 30 lines

    return {
        "file": str(p),
        "name": p.name,
        "ext": p.suffix.lower(),
        "exists": exists,
        "size_bytes": size_bytes,
        "lines": line_count,
        "modified": modified,
        "preview": preview,
    }


def build_metadata_list(read_files_output: Any) -> List[dict]:
    """
    Accepts the raw payload from a tool (e.g., read_files) or a plain list of paths/dicts.
    - If it's a dict, prefer 'summary', then 'files', then 'results'.
    - If it's a list, map each entry through format_file_metadata.
    """
    items: Sequence[Any]
    if isinstance(read_files_output, dict):
        items = (
            read_files_output.get("summary")
            or read_files_output.get("files")
            or read_files_output.get("results")
            or []
        )
    elif isinstance(read_files_output, (list, tuple)):
        items = read_files_output
    else:
        items = []

    out: List[dict] = []
    for it in items:
        try:
            out.append(format_file_metadata(it))
        except Exception as e:
            print(f"[cli] ⚠️  Skipped item due to error: {e!r} — item={repr(it)[:200]}")
    return out


def _kb(x: Any) -> str:
    if isinstance(x, (int, float)):
        return f"{x/1024:.1f}"
    return ""


def bold(text: str) -> str:
    """Return bolded text for supported terminals."""
    return f"\033[1m{text}\033[0m" if USE_COLOR else text


def display_result(result: Dict[str, Any]) -> None:
    """
    Pretty-print and log a single tool step result for user and for sharing.
    """
    tool = result.get("tool", "Unknown Tool")
    status = result.get("status", StepStatus.SUCCESS)
    message = result.get("message", "")

    # Skip noisy/low-level tools unless error
    if (
        tool in {"aggregate_file_content", "llm_response_async"}
        and status == StepStatus.SUCCESS
    ):
        return

    print(f"\n🔧 Tool: {tool}")
    print(f"🗨️  Message: {message}")
    logger.info(f"\n🔧 Tool: {tool}")
    logger.info(f"🗨️  Message: {message}")

    result_payload = result.get("result")

    # If the payload includes files/summary/results, show a metadata table
    def maybe_render_files_table(container: Any, title: str) -> bool:
        metas = build_metadata_list(container)
        if not metas:
            return False
        rows = [
            [
                m.get("file", ""),
                "yes" if m.get("exists") else "no",
                _kb(m.get("size_bytes")),
                m.get("lines") if m.get("lines") is not None else "",
                m.get("modified") or "",
            ]
            for m in metas
        ]
        headers = ["File", "Exists", "Size (KB)", "Lines", "Modified"]
        print(f"\n📁 {bold(title)}:")
        table = tabulate(rows, headers=headers, tablefmt="fancy_grid")
        print(table)
        logger.info(f"\n📁 {title}:\n{table}")
        return True

    # Smart rendering by common shapes
    if isinstance(result_payload, dict):
        rendered = False
        # Prefer showing files/summary/results as tables if present
        for field in ("summary", "files", "results"):
            if field in result_payload and maybe_render_files_table(
                result_payload, field.capitalize()
            ):
                rendered = True
                break

        # Print other keys verbosely
        if not rendered:
            pp = pprint.PrettyPrinter(indent=2, width=100, compact=False)
            # Avoid dumping giant content lists raw; collapse them
            sanitized = dict(result_payload)
            for k in ("summary", "files", "results"):
                if k in sanitized and isinstance(sanitized[k], list):
                    sanitized[k] = f"<{k}: {len(sanitized[k])} items>"
            print("\n📌 Result (object):")
            print(pp.pformat(sanitized))
            logger.info(pp.pformat(sanitized))

    elif isinstance(result_payload, list):
        # Could be a list of paths or {file,content} dicts
        if not maybe_render_files_table(result_payload, "Files"):
            pp = pprint.PrettyPrinter(indent=2, width=100, compact=False)
            print("\n📌 Result (list):")
            pp.pprint(result_payload)
            logger.info(pp.pformat(result_payload))
    else:
        print("\n📌 Result:")
        print(result_payload)
        logger.info(str(result_payload))


def run_agent_loop(agent: AgentCore) -> None:
    """
    Launches the interactive CLI session for the agent.
    Handles user input, calls the agent, and displays/logs results step by step.

    Args:
        agent (AgentCore): The main agent instance to handle instructions.
    """
    print("🧠 Config Manager CLI (type 'quit' or Ctrl+C to exit)")
    while True:
        try:
            instruction = input("\n📝 Instruction: ").strip()
            if instruction.lower() in {"quit", "exit"}:
                print("👋 Goodbye!")
                break

            logger.info("=" * 50)
            logger.info(f"User Instruction: {instruction}")

            tool_results = agent.handle_instruction(instruction)

            print("\n--- Results ---")
            logger.info("\n--- Results ---")

            for i, result in enumerate(tool_results.results):
                display_result(result.model_dump(mode="json"))
                logger.info(
                    "RESULT_JSON for step %d:\n%s", i, result.model_dump_json(indent=2)
                )

            logger.info("=" * 50)

        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            logger.error(f"Error in agent loop: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")


def main():
    """
    CLI entry point for the agent-powered Config Manager.
    Handles command-line arguments, model selection, and runs interactive or one-off mode.
    """
    try:
        local_model_names = get_local_model_names(MODEL_CONFIGS_YAML_FILE)
        api_model_names = get_api_model_names()
        local_models = get_local_models_and_help(MODEL_CONFIGS_YAML_FILE)
        api_models = get_api_models_and_help()
    except Exception as e:
        logger.error("Failed to load model information", exc_info=True)
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

    # Safeguard: ensure exactly one model
    chosen_models = [x for x in [args.local_model, args.api_model] if x]
    if len(chosen_models) > 1:
        parser.error(
            "Please specify only one model (local or cloud), not multiple at once."
        )
    if not chosen_models:
        parser.error(
            "Please specify one model: --local-model (local) or --api-model (cloud)."
        )

    # Instantiate agent
    try:
        agent = AgentCore(
            local_model_name=args.local_model, api_model_name=args.api_model
        )
    except Exception as e:
        logger.error(f"Failed to initialize AgentCore: {e}", exc_info=True)
        sys.exit(1)

    # One-off or interactive mode
    if args.once:
        instruction = args.once.strip()
        try:
            logger.info("=" * 50)
            logger.info(f"User Instruction: {instruction}")

            tool_results = agent.handle_instruction(instruction)
            print("\n--- Results ---")
            logger.info("\n--- Results ---")

            for i, result in enumerate(tool_results.results):
                display_result(result.model_dump(mode="json"))
                logger.info(
                    "RESULT_JSON for step %d:\n%s",
                    i,
                    json.dumps(
                        result.model_dump(mode="json"), indent=2, ensure_ascii=False
                    ),
                )

            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Error during instruction execution: {e}", exc_info=True)
            print(f"❌ Error: {e}")
    else:
        run_agent_loop(agent)


if __name__ == "__main__":
    main()
