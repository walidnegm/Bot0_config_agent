import argparse
from agent.core import AgentCore

def run_agent_loop():
    agent = AgentCore()

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
                tool = result.get("tool", "Unknown Tool")
                message = result.get("message", "")
                print(f"✔ {tool} → {message}")

                result_payload = result.get("result")
                if isinstance(result_payload, dict):
                    for k, v in result_payload.items():
                        if isinstance(v, list):
                            for item in v:
                                print(f"  - {item}")
                        else:
                            print(f"  {k}: {v}")

        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Agent-powered CLI for natural language instructions")
    parser.add_argument("--once", type=str, help="Run a single instruction and exit")
    args = parser.parse_args()

    if args.once:
        instruction = args.once.strip()
        agent = AgentCore()
        try:
            results = agent.handle_instruction(instruction)
            for result in results:
                print(f"{result['tool']} → {result['message']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        run_agent_loop()

if __name__ == "__main__":
    main()

