# Bot0 Config Agent

## 🚀 Quick Start

### 1. **Set Environment Variables**

```bash
export HF_TOKEN=your_token
export GITHUB_TOKEN=your_github_token
export OPENAI_API_KEY=your_openai_key
```

---

### 2. **Create a Virtual Environment & Activate**

Install dependencies with `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 3. **Download Model for Local Use**

Login to Hugging Face and prepare the model directory:

```bash
huggingface-cli login
mkdir -p model/models--meta-llama--Meta-Llama-3-8B-Instruct
cd model/models--meta-llama--Meta-Llama-3-8B-Instruct
```

Download the model using the provided script:

```bash
python downloadllama.py
```

---

### 4. **Run the CLI**

Change to your project directory and run example commands:

```bash
cd ~/../Bot0_config_agent

python -m agent.cli --once "where are my model files" --openai
python -m agent.cli --once "where are my python files in agents dir" --openai
python -m agent.cli --once "count files and directory size" --openai
```

---

## How Does It Work
---
### 🧩 Step-by-Step Sequence (High-level)

1. **User launches CLI and enters an instruction**
2. `cli.py` receives input and calls `AgentCore.handle_instruction(instruction)`
3. `AgentCore` calls `Planner.plan(instruction)`

   * Planner builds a prompt describing all available tools, adds the user instruction, and sends it to the LLM
   * LLM returns a JSON array: plan of tool calls with params
   * Planner parses/validates the JSON plan and returns it
4. `AgentCore` sends the plan to `ToolExecutor.execute_plan(plan)`

   * Executor runs each step (tool call) in order, substituting `<prev_output>` if needed
   * Loads the actual tool function via `ToolRegistry` and executes it
   * Collects the results
5. `cli.py` receives the results and displays them to the user

---

### 🗂️ **Diagram**

```
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
```

---

## 🛠️ How to Add a New Tool

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

---

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

---

### 3. **(Optional) Test the Tool**

* Start the CLI and try an instruction that should trigger your tool, e.g.:

  ```bash
  python agent/cli.py --once "Where are my project files"
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

---
## Local LLM Models

### Hardware Requirements: Desktops/Towers

**Recommended models:**
- `meta-llama/Meta-Llama-3-8B-Instruct`
- Or similar 7B+ parameter models, such as Qwen (Alibaba), Gemma (Google), DeepSeek, etc.

---

### More Laptop-Friendly Choices

As **8GB dedicated VRAM is the de facto upper limit for most of today’s gaming laptops**, we recommend **4-bit quantized small models** from Alibaba, Alphabet, Deepseek, Meta, and Microsoft to achieve the best balance of speed, performance, quality, ease of use, and cost—**without running out of memory (OOM)**.

- Best suited for laptops with **4GB–8GB dedicated VRAM**
- **GPTQ** and **AWQ quantization** are state-of-the-art methods that preserve most of the full-size model’s quality, though they are slightly more complex to implement.
- **Cost & Licensing:**
    - **Meta** models are free for research and prototyping, but have restrictions on commercial use.
    - **Microsoft’s Phi** models are completely free, even for commercial use (MIT license).
    - **Alibaba’s Qwen** and **Google’s Gemma** models are free for most uses under permissive open-source licenses (Apache 2.0), with only minor restrictions (e.g., attribution, NOTICE file).

---

---

| Model                                         | Parameters | Quantization | VRAM (GB) | Overall Quality vs. Mid-Size Unquantized Models        | Ease of Installation        | License                                 |
|-----------------------------------------------|------------|--------------|-----------|--------------------------------------------------------|----------------------------|-----------------------------------------|
| **Qwen3-4B-AWQ**                              | 4B         | AWQ 4-bit    | 2.5–4     | strong reasoning/coding but **Not instruct/chat model**  | High (vLLM, Ollama)         | Apache 2.0                              |
| **Qwen3-1.7B-Instruct-GPTQ**                  | 1.7B       | GPTQ 4-bit   | 1.5–2.5   | Lower, good for simple tasks                           | High (auto-gptq, Ollama)    | Apache 2.0                              |
| **Gemma-2-2B-it-GPTQ**                        | 2B         | GPTQ 4-bit   | 1.5–2.5   | Slightly below, efficient                              | High (gptqmodel, Ollama)    | Apache 2.0                              |
| **TheBloke/deepseek-coder-1.3b-instruct-GPTQ**| 1.3B       | GPTQ 4-bit   | 1–1.5     | Lightweight code model, surprisingly capable           | High (auto-gptq, LM Studio) | Apache 2.0                              |
| **Phi-3.5-mini-instruct-AWQ**                 | 3.8B       | AWQ 4-bit    | 2.5–3.5   | Comparable, excels in math/RAG                         | High (vLLM, LM Studio)      | MIT                                     |
| **Llama-3.2-3B-Instruct-GPTQ**                | 3B         | GPTQ 4-bit   | 3.5–4.5   | Comparable, efficient                                  | High (auto-gptq, LM Studio) | Llama-3.2 (research/limited commercial) |

---

> **Note:**  
> `Qwen3-4B-Instruct` would be a much better choice for agents/chat, but no quantized version is currently available. You would need to manually quantize it if desired.

---

### How to Download & Test

**All model names, IDs, licenses, etc. are in the `models.yaml` file.**

**To download all models:**
```sh
python -m downloaders/download_all_models.py
```

**To test the GPTQ quantized models:**
```sh
python test_gptq_quant_models.py
```
---
## How to Use Logger
In the module,
``` python
import logging
import logging_config

logger = logging.getLogger(__name__)
```
Then you can use logger.info, logger.debug, logger.error, etc. Logging files will be automaticaly saved in logs folder.

Note: you only need to import logging_config (custom .py in the project) at the entry point. Lower level modules only need to import logging.