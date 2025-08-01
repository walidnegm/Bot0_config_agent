# test_llm_prompt.py
from agent.llm_manager import LLMManager


def main():
    llm = LLMManager()

    print("\n🧠 LLM Test CLI")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("📝 Prompt: ").strip()
            if prompt.lower() in ("exit", "quit"):
                print("👋 Exiting.")
                break

            response = llm.generate(prompt=prompt, temperature=0.1)
            print(f"\n📤 LLM Response:\n{response}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
