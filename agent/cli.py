import argparse
from agent.core import AgentCore

def main():
    parser = argparse.ArgumentParser(description="Config Manager CLI")
    parser.add_argument("instruction", type=str, nargs="+", help="Natural language instruction")
    args = parser.parse_args()

    instruction = " ".join(args.instruction)
    agent = AgentCore()

    try:
        results = agent.run(instruction)
        print("\n--- Result ---")
        for result in results:
            print(f"✔ {result['tool']} → {result['message']}")
            if result["tool"] == "list_project_files":
                files = result["result"].get("files", [])
                for f in files:
                    print(f"  - {f}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

