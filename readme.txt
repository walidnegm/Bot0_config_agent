SET THESE ENVIRONMENT VARIABLES
-------------------------------
export HF_TOKEN=your_token
export GITHUB_TOKEN=your_github_token
export OPENAI_API_KEY=your_openai_key

CREATE A VIRTUAL ENVIRONMNET & ACTIVATE
---------------------------------------
use requirements.txt

DOWNLOAD MOEDL TO USE LOCALLY
-------------------------------
huggingface-cli login
mkdir -p model/models--meta-llama--Meta-Llama-3-8B-Instruct
cd model/models--meta-llama--Meta-Llama-3-8B-Instruct

Run script to download the model - downloadllama.py

To Run the CLI
--------------
cd ~/../Bot0_config_agent
python -m agent.cli --once "where are my model files" --openai ~/projects/Bot0_config_agent
python -m agent.cli --once "where are my python files in agent dir" --openai


Step-by-Step Sequence (High-level)
----------------------------------
User launches CLI and enters an instruction
1. cli.py receives input and calls AgentCore.handle_instruction(instruction)
2. AgentCore calls Planner.plan(instruction)
- Planner builds a prompt describing all available tools, adds the user instruction, and sends it to the LLM
- LLM returns a JSON array: plan of tool calls with params
- Planner parses/validates the JSON plan and returns it
4. AgentCore sends the plan to ToolExecutor.execute_plan(plan)
- Executor runs each step (tool call) in order, substituting <prev_output> if needed
- Loads the actual tool function via ToolRegistry and executes it
- Collects the results
5. cli.py receives the results and displays them to the user

Diagram
User
 |
 v
cli.py
 |---> [input] ----> AgentCore.handle_instruction()
                       |
                       v
                Planner.plan()
                       |
               [LLM (local/OpenAI)]
                       |
              <--[JSON tool plan]---
                       |
                       v
                ToolExecutor.execute_plan()
                       |
           [for each tool call in plan:]
                       |
             ToolRegistry.get_function()
                       |
             [run tool function, collect result]
                       |
           <---------- results (list) ---------
 |
 v
cli.py
 |
[format/display output]
 |
User sees results


🛠️ How to Add a New Tool
-------------------------
Absolutely! Here’s a **concise step-by-step guide** for adding a new tool to your agent, in a format ready for inclusion in your `README.md`:

### 1. **Implement Your Tool Function**

* Write a Python function in the `tools/` directory.
* The function should accept keyword arguments (`**kwargs` or named args) and return a `dict` with `status`, `message`, and (optionally) a `result` field.

**Example:**

```python
# tools/hello_tool.py

def hello_tool(**kwargs):
    name = kwargs.get("name", "World")
    return {
        "status": "ok",
        "message": f"Hello, {name}!"
    }
```

### 2. **Register the Tool in `tool_registry.json`**

* Add an entry for your tool in `tools/tool_registry.json`.
* Set the `import_path` to the Python path of your function.
* Define parameters as a JSON schema object.

**Example:**

```json
"hello_tool": {
  "description": "Greets the specified name.",
  "import_path": "tools.hello_tool.hello_tool",
  "parameters": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Name to greet."
      }
    },
    "required": []
  }
}
```
### 3. **(Optional) Test the Tool**

* Start the CLI and try an instruction that should trigger your tool, e.g.:

  ```
  python agent/cli.py --once "say hello to Alice"
  ```
* If you see validation errors, check your function name and parameters in `tool_registry.json`.

---

### 4. **That's It!**

* The agent will automatically discover and use your tool.
* The LLM will see your tool’s name, parameters, and description in its prompt.

---

**Tips:**

* Use clear descriptions and parameter names—they help the LLM use your tool correctly!
* Return errors as `{"status": "error", "message": "description of error"}` for consistency.
