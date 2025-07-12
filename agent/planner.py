# agent/planner.py

import json
import re
from agent.llm_manager import LLMManager
from agent.prompt_builder import PromptBuilder

class Planner:
    def __init__(self, registry):
        self.registry = registry
        self.llm = LLMManager()
        self.prompt_builder = PromptBuilder(registry)

    def extract_first_json_array(self, text: str) -> str:
        """
        Extracts the first valid JSON array from a potentially noisy LLM response.
        Strips common wrappers first.
        """
        # Strip Markdown/code blocks and labels
        text = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*(Here\'s the plan:|Output:)\s*', '', text, flags=re.IGNORECASE)
        text = text.strip()

        # Find the outermost array: Look for [ ... ] with balanced content
        # Use a better pattern for arrays of objects
        match = re.search(r'\[\s*(?:\{.*?\}\s*,?\s*)*\]', text, re.DOTALL)
        if not match:
            raise ValueError(f"❌ No valid JSON array found in response: {text}")
        return match.group(0)

    def plan(self, instruction: str):
        tools = self.registry.get_all()
        prompt = self.prompt_builder.build_prompt(instruction, tools)

        print(f"\n[PromptBuilder] Prompt:\n{prompt}\n")
        response = self.llm.generate(prompt)
        print(f"[LLMManager] Raw Output:\n{response}\n")

        try:
            json_block = self.extract_first_json_array(response)
            return json.loads(json_block)
        except Exception as e:
            raise ValueError(f"❌ Failed to parse tools JSON: {e}")
