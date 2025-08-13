from tools.tool_registry import ToolRegistry
from agent.prompt_builder import PromptBuilder

def main():
    instruction = "scan and echo project files"  # or change as needed

    registry = ToolRegistry()
    tools = registry.get_all()

    builder = PromptBuilder(registry)
    prompt = builder.build_prompt(instruction, tools)

    print("\n--- Generated Prompt ---\n")
    print(prompt)

if __name__ == "__main__":
    main()

