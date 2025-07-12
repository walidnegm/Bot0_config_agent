class PromptBuilder:
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def build_prompt(self, instruction: str, tools: list) -> str:
        tool_lines = []
        for tool in tools:
            name = tool["name"]
            desc = tool["description"]
            props = tool["parameters"]["properties"]
            args = ", ".join(f'{key}: {val["type"]}' for key, val in props.items())
            tool_lines.append(f"- {name}({args}): {desc}")

        tools_block = "\n".join(tool_lines)

        prompt = f"""You are a precise tool-calling agent. You have access to the following tools:

{tools_block}

Your response MUST be ONLY a single valid JSON array of objects, starting directly with [ and ending with ].
Do NOT add ANY text, explanations, code blocks, backticks (```), Markdown, or extra characters before, after, or around the JSON.
Do NOT wrap in "json" or any labels. Output raw JSON only.

Each item in the array should be an object like {{"tool": "tool_name", "params": {{"arg1": "value1", "arg2": "value2"}}}}.
For multi-step instructions, use sequential items in the array and reference previous outputs with "<prev_output>" in params.

Example of correct output for a single-step instruction:
[{{"tool": "list_project_files", "params": {{"root": "path/to/project"}}}}]

Example for multi-step (e.g., list then echo):
[{{"tool": "list_project_files", "params": {{"root": "path/to/project"}}}}, {{"tool": "echo_message", "params": {{"message": "<prev_output>"}}}}]

Now, respond only with the JSON array for this instruction: {instruction}
"""
        return prompt
