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

### 3. **Test the Tool**

* Start the CLI and try an instruction that should trigger your tool, e.g.:

  ```bash
  python agent/cli.py --show-models-help

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
## How to Use Logger
In the module,
``` python
import logging
import logging_config

logger = logging.getLogger(__name__)
```
Then you can use logger.info, logger.debug, logger.error, etc. Logging files will be automaticaly saved in logs folder.


Note: you only need to import logging_config (custom .py in the project) at the entry point. Lower level modules only need to import logging.

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
### Where to Save Local LLMs

- **If the model is downloaded via Hugging Face's `transformers` or `huggingface_hub`:**
  - Let it save to the default `~/.cache/huggingface/` directory.
  - ✅ **Do NOT manually specify the download or load location.**
  - All standard `AutoModel` and `snapshot_download` calls will reuse this shared cache automatically.

- **If the model is manually cloned or downloaded (e.g., GGUF, GPTQ, AWQ models):**
  - Save it in the `models/` directory at the project root and update the models_config.yaml file.
  - This directory is already listed in `.gitignore`, so models will not be tracked in Git.
  - Loader functions (e.g., `LLMManager`) will resolve `model_path` relative to the project root.

--- 
NOT ON QUANTIZED MODELS
-----------------------
🌀 What You Tried
Built a Python 3.10 virtual environment to compile auto-gptq, because GPTQ quantized models failed in Python 3.12.

Attempted to compile auto-gptq with CUDA support under WSL2.

Resolved:

Missing nvcc

Missing .so symbols (e.g., ncclCommRegister)

LD_LIBRARY_PATH and PATH adjustments

Ultimately hit architectural compatibility issues (WSL2 / driver quirks) even after proper CUDA tooling was installed.

🎯 Where You Landed
🤝 transformers and optimum now natively support GPTQ models (e.g., TheBloke/...-GPTQ).

You no longer need AutoGPTQForCausalLM or to build auto-gptq manually.

AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True) is enough — no more forks, no more patching, and no more fragile builds.

✅ Final State
You're back to a cleaner, stable approach:

Use the mainline Hugging Face stack.

One loader path.

Quantized or not, same interface.

No more manual compilation or unmaintained forks

---
## Depedencies
How to install gptqmodel:
pip install --no-build-isolation gptqmodel
---

## 🔧 Using LLMManager

The `LLMManager` class loads a quantized model (GPTQ, GGUF, or AWQ) and exposes a `.generate()` method for local inference.

It expects a `models.yaml` config file like this:

```yaml
gptq_llama:
  model_path: models/llama2-7b-gptq
  loader: gptq
  torch_dtype: float16
  temperature: 0.7
```
Example Usage:
```
from agent.llm_manager import LLMManager

# Load a model named 'gptq_llama' from models.yaml
llm = LLMManager(model_name="gptq_llama")

# Call the model to generate a tool-calling JSON
response = llm.generate("summarize the config files")
print(response)
```

## Loader and Config
We standardize on 4 primary loaders ("backends") for managing local model inference.

### Loaders (Backends) — “How to load and run the model”
| Loader           | Backend Library            | Description                                                                |
| ---------------- | -------------------------- | -------------------------------------------------------------------------- |
| `transformers`   | 🤗 `transformers`          | Standard FP16/BF16 models and basic quantized `.bin`/`.safetensors` models |
| `gptq`           | `gptqmodel` (not autogptq) | Custom GPTQ loader with faster low-level inference                         |
| `awq`            | `autoawq`                  | 4-bit quantized model loader using AWQ with fused optimizations            |
| `llama_cpp`      | `llama-cpp-python`         | GGUF-based model inference using llama.cpp backend                         |
| `vllm` (planned) | `vllm` engine              | Fast tokenizer-aware KV-caching runtime (batch optimized)                  |


### Config (Per-model settings) — “What to do for this model”
| Config Key            | Example Values                | Purpose                                                 |
| --------------------- | ----------------------------- | ------------------------------------------------------- |
| `model_id`            | `TheBloke/Llama-2-7B-AWQ`     | Logical identifier for tracking the model               |
| `model_path`          | `./models/llama2.Q4_K_M.gguf` | Local path to model weights (for AWQ, GGUF, GPTQ, etc.) |
| `torch_dtype`         | `float16`, `bfloat16`         | Precision to load weights and run inference             |
| `device`              | `cuda`, `cpu`, `auto`         | Device assignment                                       |
| `use_safetensors`     | `true/false`                  | Choose `.safetensors` over `.bin` if both exist         |
| `offload_folder`      | `./offload`                   | Optional CPU offload folder (e.g. LLaMA-3)              |
| `trust_remote_code`   | `true`                        | Required for custom models like LFM2 or fine-tuned ones |
| `quantization_config` | `{bits: 4, group_size: 128}`  | (Optional) metadata for tracking quantization strategy  |


### Loader Reference
| Loader         | What it Means                       | Backend           | Notes                                                |
| -------------- | ----------------------------------- | ----------------- | ---------------------------------------------------- |
| `transformers` | Hugging Face standard model loading | `transformers`    | Supports full precision and quantized formats        |
| `gptq`         | GPTQ quantized model (4-bit)        | `gptqmodel`       | Uses `.safetensors`; does not use transformers       |
| `awq`          | AWQ 4-bit quantized model           | `autoawq`         | Fast AWQ loader; not Hugging Face-compatible         |
| `llama_cpp`    | GGUF model for llama.cpp            | `llama_cpp.Llama` | Loads `.gguf`; extremely efficient for local CPU/GPU |
| `vllm` (WIP)   | Token streaming with KV caching     | `vllm` engine     | Not yet integrated, but future option for batching   |

## Local LLM Loader Testing Progress

### Models Saved Locally in Project Directory
- `qwen3_1_7b_instruct_gptq` — saved to `project/models/`; text generation tested and working properly.
- `qwen3_4b_awq` — saved to `project/models/`; text generation tested and working, but it repeats itself.
- `deepseek_coder_1_3b_gptq` — saved to `project/models/`; text generation tested and working partially but it repeats itself.
- `gemma_2_2b_gptq` — saved to `project/models/`; text generation tested and working properly.
- `llama_2_7b_chat_gptq` — saved to `project/models/`; text generation tested but OOM - too large for 4.1GB VRAM.
- `llama_3_2_3b_gptq` — saved to `project/models/`; text generation tested and working properly.
- `phi_3_5_mini_awq` — saved to `project/models/`; text generation tested but not working!!!
- `tinyllama_1_1b_chat_gguf` — saved to `project/models/`, text generation tested and working properly

### Models Remaining in Default Huggingface Cache
- `lfm2_1_2b` (LiquidAI/LFM2-1.2B) - saved to .cache; can't get it to work for text generation
- `llama3_8b` (meta-llama/Meta-Llama-3-8B-Instruct); too large for 4GB VRAM

If you are using Linux filing system, the default .cache location is usually at:<br>
~/.cache/huggingface/hub/models--....

---
### Summary
| Model Name                 | Status          | Notes                                                    |
| -------------------------- | ----------------| ---------------------------------------------------------|
| `qwen3_1_7b_instruct_gptq` | ✅ Working      | Text generation tested and working properly              |
| `qwen3_4b_awq`             | ⚠️ Partial      | Repeats itself; text generation works but needs guardrail|
| `deepseek_coder_1_3b_gptq` | ⚠️ Partial      | Repeats itself; text generation works but needs guardrail|
| `gemma_2_2b_gptq`          | ✅ Working      | Text generation tested and working properly              |
| `llama_2_7b_chat_gptq`     | ❌ Failed (OOM) | Too large for 4.1GB VRAM                                 |
| `llama_3_2_3b_gptq`        | ✅ Working      | Text generation tested and working properly              |
| `phi_3_5_mini_awq`         | ❌ Failed       | Text generation not working at all                       |
| `tinyllama_1_1b_chat_gguf` | ✅ Working      | Text generation tested and working properly              |
--- 

