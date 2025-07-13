# agent/prompt_builder.py

class PromptBuilder:
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def build_prompt(self, instruction: str, tools: dict) -> str:
        tool_lines = []
        for name, tool in tools.items():  # ✅ Fixed: use .items() for dict
            desc = tool.get("description", "No description provided.")
            props = tool.get("parameters", {}).get("properties", {})
            args = ", ".join(f'{key}: {val.get("type", "unknown")}' for key, val in props.items())
            tool_lines.append(f"- {name}({args}): {desc}")

        tools_block = "\n".join(tool_lines)

        prompt = f"""You are a precise tool-calling agent. You have access to the following tools:

{tools_block}

Your ONLY output must be a single valid JSON array of objects, without any preamble or explanation.

Format strictly as follows:
[
  {{
    "tool": "tool_name",
    "params": {{
      "arg1": "value1",
      "arg2": "value2"
    }}
  }},
  ...
]

Do NOT include Markdown formatting, comments, code blocks (no ```), or labels like "json".
Just return the raw JSON array.

If no tool is relevant, return an empty array: []

For multi-step tasks, return multiple tool calls in sequence. You may refer to previous tool outputs using the string "<prev_output>".

Examples:

Single step:
[
  {{
    "tool": "list_project_files",
    "params": {{
      "root": "."
    }}
  }}
]

Two steps:
[
  {{
    "tool": "list_project_files",
    "params": {{
      "root": "."
    }}
  }},
  {{
    "tool": "echo_message",
    "params": {{
      "message": "<prev_output>"
    }}
  }}
]

Instruction: {instruction}
"""
        return prompt

