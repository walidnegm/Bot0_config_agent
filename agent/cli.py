# cli.py
# 🧠 Natural Language CLI for Config Manager
#
# Usage examples:
#   Run interactively with local model:
#     python agent/cli.py
#
#   Run interactively with OpenAI GPT backend:
#     python agent/cli.py --openai
#
#   Run one-off command with local model:
#     python agent/cli.py --once "where are my model files"
#
#   Run one-off command with OpenAI GPT backend:
#     python agent/cli.py --once "where are my model files" --openai

import os
import argparse
import datetime
from tabulate import tabulate
from agent.core import AgentCore


def format_file_metadata(file_path):
    try:
        size = os.path.getsize(file_path)
        created_ts = os.path.getctime(file_path)
        size_str = f"{size / 1024:.1f} KB" if size < 1e6 else f"{size / (1024 * 1024):.1f} MB"
        created_str = datetime.datetime.fromtimestamp(created_ts).strftime('%Y-%m-%d %H:%M')
        return [file_path, size_str, created_str]
    except Exception:
        return [file_path, "?", "?"]


def display_result(result):
    tool = result.get("tool", "Unknown Tool")
    message = result.get("message", "")
    print(f"\n\033[1m🔧 Tool:\033[0m {tool}")
    print(f"\033[1m🗨️  Message:\033[0m {message}")

    # Check inside result.result for file-related fields
    result_payload = result.get("result")
    if isinstance(result_payload, dict):
        for field in ("matches", "files", "results"):
            items = result_payload.get(field)
            if isinstance(items, list) and items:
                rows = [format_file_metadata(path) for path in items]
                print(f"\n📁 \033[1m{field.capitalize()}:\033[0m")
                print(tabulate(rows, headers=["Path", "Size", "Created"], tablefmt="fancy_grid"))

        for k, v in result_payload.items():
            if k in {"matches", "files", "results"}:
                continue
            print(f"📌 {k}: {v}")


def run_agent_loop(use_openai=False):
    agent = AgentCore(use_openai=use_openai)
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
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Agent-powered CLI for natural language instructions")
    parser.add_argument("--once", type=str, help="Run a single instruction and exit")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI GPT backend instead of local model")
    args = parser.parse_args()

    if args.once:
        instruction = args.once.strip()
        agent = AgentCore(use_openai=args.openai)
        try:
            results = agent.handle_instruction(instruction)
            for result in results:
                display_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        run_agent_loop(use_openai=args.openai)


if __name__ == "__main__":
    main()

