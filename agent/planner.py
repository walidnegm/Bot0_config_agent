import json
import os
import re
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError
from tools.tool_registry import ToolRegistry
from agent import llm_openai  # OpenAI used only if enabled
from agent.intent_classifier_core import classify_describe_only


class ToolCall(BaseModel):
    tool: str
    params: Dict[str, Any]


class Planner:
    def __init__(self, use_openai: bool = False):
        self.tool_registry = ToolRegistry()
        self.use_openai = use_openai

        if self.use_openai:
            print("[Planner] ⚙️ Using OpenAI backend")
        else:
            print("[Planner] ⚙️ Using local model")
            from agent.llm_manager import LLMManager  # only if local model is used
            self.llm_manager = LLMManager()

    def plan(self, instruction: str):
        tools = self.tool_registry.get_all()
        print("[Planner] 🔧 Retrieved tools:", tools)

        prompt = self._build_prompt(instruction, tools)
        print("\n[PromptBuilder] 📜 Prompt:\n" + prompt)

        if self.use_openai:
            llm_output = llm_openai.generate(prompt)
        else:
            llm_output = self.llm_manager.generate(prompt)

        print("\n[Planner] 📤 LLM raw response:\n" + repr(llm_output))

        try:
            extracted_json = self._extract_json_from_response(llm_output)
            print("\n[Planner] ✅ Extracted JSON array:\n", extracted_json)

            if extracted_json.strip() == "[]":
                print("[Planner] 🤖 No tool call. Checking for intent fallback…")
                intent = classify_describe_only(instruction)
                if intent == "describe_project":
                    print("[Planner] 🧠 Intent = describe_project → injecting filtered file summary plan.")
                    return self._build_filtered_project_summary_plan()

                print("[Planner] 🤷 No matched intent. Falling back to natural response.")
                answer = llm_openai.generate(instruction) if self.use_openai else self.llm_manager.generate(instruction)
                return [{
                    "tool": "llm_response",
                    "status": "ok",
                    "message": answer,
                    "result": {"text": answer}
                }]

            raw_tool_calls = json.loads(extracted_json)
            validated_calls: List[ToolCall] = []
            for i, item in enumerate(raw_tool_calls):
                try:
                    validated = ToolCall(**item)
                    validated_calls.append(validated)
                except ValidationError as ve:
                    print(f"[Planner] ❌ Validation error in item {i}:\n{ve}\n")

            print("\n[Planner] 🔍 Final planned tools:")
            for call in validated_calls:
                print(f"  → {call.tool} with params {call.params}")

            return [call.dict() for call in validated_calls]

        except Exception as e:
            print("\n[Planner] ❌ Failed to parse tools JSON:\n", repr(llm_output))
            raise ValueError(f"❌ Failed to parse tools JSON: {e}")

    def _build_filtered_project_summary_plan(self) -> List[Dict[str, Any]]:
        files_to_read = []
        for root, _, files in os.walk("."):
            if any(skip in root for skip in ["venv", "__pycache__", "models"]):
                continue
            for fname in files:
                if not fname.endswith((".py", ".md", ".toml")):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    if os.path.getsize(fpath) <= 10_000:
                        files_to_read.append(fpath)
                except OSError:
                    continue

        files_to_read = sorted(files_to_read)[:5]

        plan = []
        step_refs = []
        for idx, fpath in enumerate(files_to_read):
            plan.append({"tool": "read_file", "params": {"path": fpath}})
            step_refs.append(f"<step_{idx}>")

        plan.append({
            "tool": "aggregate_file_content",
            "params": {"steps": step_refs}
        })

        plan.append({
            "tool": "llm_response",
            "params": {
                "prompt": "Summarize this project: <prev_output>"
            }
        })

        return plan

    def _build_prompt(self, instruction, tools):
        tool_descriptions = []
        for name, meta in tools.items():
            params = meta["parameters"]["properties"]
            param_desc = ", ".join([f"{k}: {v['type']}" for k, v in params.items()])

            usage_hint = ""
            if name == "find_file_by_keyword":
                usage_hint = " Use this for vague file searches like 'llama', 'model', 'snapshot'."
            elif name == "list_project_files":
                usage_hint = " Use this to list all files in a folder."

            tool_descriptions.append(f"- {name}({param_desc}): {meta['description']}.{usage_hint}")

        tools_block = "\n".join(tool_descriptions)

        prompt = (
            "You are a precise tool-calling agent. You have access to the following tools:\n\n"
            + tools_block + "\n\n"
            "Your ONLY output must be a single valid JSON array of objects, without any preamble or explanation.\n\n"
            "Format strictly as follows:\n"
            "[\n"
            "  {\n"
            "    \"tool\": \"tool_name\",\n"
            "    \"params\": {\n"
            "      \"arg1\": \"value1\",\n"
            "      \"arg2\": \"value2\"\n"
            "    }\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "Do NOT include Markdown formatting, comments, code blocks (no ```), or labels like \"json\".\n"
            "Just return the raw JSON array.\n\n"
            "If no tool is relevant, return an empty array: []\n\n"
            "For multi-step tasks, return multiple tool calls in sequence. You may refer to previous tool outputs using the string \"<prev_output>\" or \"<step_n>\".\n\n"
            "💡 Tip: If the instruction involves finding or listing files, follow up with the `echo_message` tool to clearly show the matched files to the user.\n\n"
            f"Instruction: {instruction}"
        )
        return prompt

    def _extract_json_from_response(self, text: str) -> str:
        text = text.strip().replace("```json", "").replace("```", "")
        if text == "[]":
            return "[]"

        matches = re.findall(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list) and all("tool" in x and "params" in x for x in parsed):
                    return json.dumps(parsed)
            except Exception:
                continue

        raise ValueError("No valid JSON array of tool calls found in LLM response.")

