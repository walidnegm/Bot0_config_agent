import json
import os
import re
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError
from tools.tool_registry import ToolRegistry
from agent import llm_openai
from agent.intent_classifier_core import classify_describe_only
from agent.llm_manager import (
    LLMManager,
)  # only if local model is used


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

            self.llm_manager = LLMManager()

        self.param_aliases = {
            "file_path": "path",
            "filename": "path",
            "filepath": "path",
        }

    def plan(self, instruction: str) -> List[Dict[str, Any]]:
        instruction = instruction.strip()

        tools = self.tool_registry.get_all()
        tool_names = ", ".join(sorted(tools.keys()))
        system_msg = (
            "Return a valid JSON array of tool calls. "
            'Format: [{ "tool": "tool_name", "params": { ... } }]. '
            f"The key must be 'tool' (not 'call'), and 'tool' must be one of: {tool_names}. "
            "Use exactly the parameter names shown in the tool spec. "
            'For example: use "path" (not "file_path") for read_file. '
            "For general knowledge or definitions, return []. Do NOT invent new tool names or use placeholders like 'path/to/file'. "
            "If a file must be found first, use `list_project_files` or `find_file_by_keyword` first, then refer to their output."
        )

        # ✅ Run intent classification before using the LLM's tool plan
        intent = classify_describe_only(instruction, use_openai=self.use_openai)
        print(f"[Planner] 🧠 Parsed intent: {intent}")
        if intent == "describe_project":
            print(
                "[Planner] 🔁 Overriding tool plan — injecting describe_project summary plan"
            )
            return self._build_filtered_project_summary_plan()

        prompt = self._build_prompt(instruction, tools)
        print("\n[PromptBuilder] 📜 Prompt:\n" + prompt)
        # print("[Planner] 🔧 Retrieved tools:", self.tool_registry.get_all())

        if self.use_openai:
            llm_output = llm_openai.generate(prompt)
        else:
            llm_output = self.llm_manager.generate(
                prompt, system_prompt=system_msg, max_new_tokens=512, temperature=0.0
            )

        # 🔹 Extract JSON array from LLM output
        if isinstance(llm_output, dict):
            llm_output = llm_output.get("text") or llm_output.get("output") or ""

        print("\n[Planner] 📤 LLM raw response:\n" + repr(llm_output))

        try:
            if self.use_openai:
                extracted_json = llm_output.strip()  # OpenAI already returns clean JSON
            else:
                extracted_json = self._extract_json_from_response(llm_output)

            print("\n[Planner] ✅ Extracted JSON array:\n", extracted_json)

            raw_tool_calls = json.loads(extracted_json)
            validated_calls: List[ToolCall] = []

            for i, item in enumerate(raw_tool_calls):
                tool_name = item.get("tool")
                params = item.get("params", {})
                # 🔧 Normalize parameter keys
                for old_key, new_key in self.param_aliases.items():
                    if old_key in params and new_key not in params:
                        params[new_key] = params.pop(old_key)
                # 🔧 FIX BAD PARAMS: Remove unexpected ones

                if tool_name not in self.tool_registry.tools:
                    print(f"[Planner] ⚠️ Unknown tool: {tool_name}")
                    return [{"tool": "llm_response", "params": {"prompt": instruction}}]

                # Auto-fix common bad param
                if (
                    tool_name == "list_project_files"
                    and "files" in params
                    and "root" not in params
                ):
                    print("[Planner] ⚠️ Auto-rewriting 'files' param to 'root'")
                    params["root"] = "."
                    del params["files"]

                placeholder_path = params.get("path", "")
                if any(
                    kw in str(placeholder_path)
                    for kw in ["path/to", "your_", "placeholder"]
                ):
                    print(
                        f"[Planner] ⚠️ Placeholder path '{placeholder_path}' detected."
                    )
                    return [
                        {
                            "tool": "find_file_by_keyword",
                            "params": {"keywords": ["python"], "root": "."},
                        },
                        {
                            "tool": "echo_message",
                            "params": {"message": "<prev_output>"},
                        },
                    ]

                if any(
                    "path/to/" in str(v) or "your/" in str(v) for v in params.values()
                ):
                    print(
                        f"[Planner] ⚠️ Placeholder detected → rewriting to find_file_by_keyword + echo_message."
                    )
                    return [
                        {
                            "tool": "find_file_by_keyword",
                            "params": {"keywords": ["python", "py"], "root": "."},
                        },
                        {
                            "tool": "echo_message",
                            "params": {"message": "<prev_output>"},
                        },
                    ]
                # Only validate params after confirming the tool exists
                valid_keys = set(
                    self.tool_registry.tools[tool_name]
                    .get("parameters", {})
                    .get("properties", {})
                    .keys()
                )

                params = {k: v for k, v in params.items() if k in valid_keys}
                item["params"] = params
                print(f"[Planner] 🔄 Normalized call {i}: {item}")

                try:
                    validated = ToolCall(**item)
                    if validated.tool not in self.tool_registry.tools:
                        print(
                            f"[Planner] ⚠️ Invalid tool: {validated.tool}. Falling back to llm_response."
                        )
                        return [
                            {"tool": "llm_response", "params": {"prompt": instruction}}
                        ]
                    validated_calls.append(validated)
                except ValidationError as ve:
                    print(f"[Planner] ❌ Validation error in item {i}:\n{ve}\n")
                    return [{"tool": "llm_response", "params": {"prompt": instruction}}]

            if extracted_json.strip() == "[]" or not validated_calls:
                print("[Planner] 🤷 No valid tools matched. Using llm_response.")
                return [{"tool": "llm_response", "params": {"prompt": instruction}}]

            print("\n[Planner] 🔍 Final planned tools:")
            for call in validated_calls:
                print(f"  → {call.tool} with params {call.params}")

            return [call.dict() for call in validated_calls]

        except Exception as e:
            print(f"[Planner] ❌ Failed to parse tools JSON: {e}\n")
            return [{"tool": "llm_response", "params": {"prompt": instruction}}]

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

        plan.append({"tool": "aggregate_file_content", "params": {"steps": step_refs}})
        plan.append(
            {
                "tool": "llm_response",
                "params": {
                    "prompt": (
                        "Give a concise summary of the project based on the following files:\n\n"
                        "<prev_output>\n\nHighlight purpose, key components, and usage."
                    )
                },
            }
        )

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
            tool_descriptions.append(
                f"- {name}({param_desc}): {meta['description']}."
                f"{' ⚡ Use this to ' + meta['description'].lower() if 'count' in name or 'size' in meta['description'].lower() else ''}"
                f"{usage_hint}"
            )

        tools_block = "\n".join(tool_descriptions)
        prompt = (
            f"You are a precise tool-calling agent. You have access to the following tools:\n\n"
            f"{tools_block}\n\n"
            "Your ONLY output must be a single valid JSON array of objects, without any preamble or explanation.\n\n"
            "Format strictly as follows:\n"
            "[\n"
            "  {\n"
            '    "tool": "tool_name",\n'
            '    "params": {\n'
            '      "arg1": "value1",\n'
            '      "arg2": "value2"\n'
            "    }\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "Do NOT include:\n"
            "- Markdown formatting or code blocks (no ```)\n"
            "- JavaScript-style expressions (no '+', '?', ':', etc.)\n"
            "- Inline comments like // or #\n"
            "- Any explanation or commentary\n"
            "Only output pure JSON with static values (strings, arrays, or numbers)."
            "Just return the raw JSON array.\n\n"
            "If no tool is relevant, return an empty array: []\n\n"
            'For multi-step tasks, return multiple tool calls in sequence. You may refer to previous tool outputs using the string "<prev_output>" or "<step_n>".\n\n'
            "💡 Tip: If the instruction involves finding or listing files, follow up with the `echo_message` tool to clearly show the matched files to the user.\n\n"
            f"Instruction: {instruction}"
        )
        return prompt

    def _extract_json_from_response(self, text: str) -> str:
        def is_valid_tool_array(candidate) -> bool:
            try:
                parsed = json.loads(candidate)
                return isinstance(parsed, list) and all(
                    isinstance(x, dict) and "tool" in x and "params" in x
                    for x in parsed
                )
            except Exception:
                return False

        # 💡 For debugging: show full text before any cleanup
        print("[Planner] 🧪 Full LLM response:\n", repr(text))

        # Step 1: Strip special tokens and markdown remnants
        text = text.replace("<|startoftext|>", "")

        # Step 2: Extract only the assistant's last message
        matches = re.findall(
            r"<\|im_start\|>assistant(?:\n|\r|\r\n)(.*?)(?:\n)?<\|im_end\|>",
            text,
            flags=re.DOTALL,
        )
        if matches:
            text = matches[-1].strip()
        else:
            # Fallback: try extracting the first valid-looking JSON array
            print("[Planner] ⚠️ No assistant block found — trying raw JSON fallback.")
            json_candidates = re.findall(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
            for candidate in json_candidates:
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    continue
            raise ValueError("No valid assistant response found.")

        # Step 3: Drop anything before first `[`
        bracket_idx = text.find("[")
        if bracket_idx != -1:
            text = text[bracket_idx:]

        # Step 4: Try parsing the full block as JSON
        try:
            if is_valid_tool_array(text):
                return text
            else:
                print("[Planner] ❌ Top-level block is not a valid tool array.")
        except Exception as e:
            print(f"[Planner] ⚠️ Error while checking top-level tool array: {e}")

        # Step 5: Try fallback via regex match for any JSON array
        regex_matches = re.findall(
            r"\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*]", text
        )
        for match in regex_matches:
            if is_valid_tool_array(match):
                return match

        print("[Planner] ❌ No valid JSON tool array found in LLM response.")
        raise ValueError("No valid JSON array of tool calls found in LLM response.")
