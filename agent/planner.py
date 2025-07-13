# agent/planner.py

import json
import re
from tools.tool_registry import ToolRegistry
from agent.llm_manager import LLMManager


class Planner:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.llm_manager = LLMManager()

    def plan(self, instruction: str):
        tools = self.tool_registry.get_all()
        print("[Planner] 🔧 Retrieved tools:", tools)

        prompt = self._build_prompt(instruction, tools)
        print("\n[PromptBuilder] 🧾 Prompt:\n" + prompt)

        llm_output = self.llm_manager.generate(prompt)
        print("\n[Planner] 📤 LLM raw response:\n" + repr(llm_output))

        try:
            extracted_json = self._extract_json_from_response(llm_output)
            print("\n[Planner] ✅ Extracted JSON array:\n", extracted_json)
            return json.loads(extracted_json)
        except Exception as e:
            print("\n[Planner] ❌ Failed to parse tools JSON:\n", repr(llm_output))
            raise ValueError(f"❌ Failed to parse tools JSON: {e}")

    def _build_prompt(self, instruction, tools):
        tool_descriptions = []
        for name, meta in tools.items():
            params = meta["parameters"]["properties"]
            param_desc = ", ".join(
                [f"{k}: {v['type']}" for k, v in params.items()]
            )
            tool_descriptions.append(f"- {name}({param_desc}): {meta['description']}")

        prompt = (
            "You are a precise tool-calling agent. You have access to the following tools:\n\n"
            + "\n".join(tool_descriptions)
            + "\n\nYour ONLY output must be a single valid JSON array of objects, without any preamble or explanation.\n\n"
            "Format strictly as follows:\n[\n  {\n    \"tool\": \"tool_name\",\n    \"params\": {\n      \"arg1\": \"value1\",\n      \"arg2\": \"value2\"\n    }\n  },\n  ...\n]\n\n"
            "Do NOT include Markdown formatting, comments, code blocks (no ```), or labels like \"json\".\n"
            "Just return the raw JSON array.\n\n"
            "If no tool is relevant, return an empty array: []\n\n"
            "For multi-step tasks, return multiple tool calls in sequence. You may refer to previous tool outputs using the string \"<prev_output>\".\n\n"
            "Examples:\n\n"
            "Single step:\n[\n  {\n    \"tool\": \"list_project_files\",\n    \"params\": {\n      \"root\": \"/path/to/project\"\n    }\n  }\n]\n\n"
            "Two steps:\n[\n  {\n    \"tool\": \"list_project_files\",\n    \"params\": {\n      \"root\": \"/my/project\"\n    }\n  },\n  {\n    \"tool\": \"echo_message\",\n    \"params\": {\n      \"message\": \"<prev_output>\"\n    }\n  }\n]\n\n"
            f"Instruction: {instruction}"
        )
        return prompt

    def _extract_json_from_response(self, text):
        """
        Extracts the first valid JSON array from LLM output using regex,
        avoiding markdown/code blocks and surrounding instructions.
        """
        text = text.strip().replace("```json", "").replace("```", "")
        match = re.search(r'\[\s*(?:\{.*?\}\s*,?\s*)+\]', text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON array found in the LLM response.")
        return match.group(0)

