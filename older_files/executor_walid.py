from typing import List, Dict, Any
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools.tool_registry import ToolRegistry
from llama_cpp import Llama
from agent_models.agent_models import StepStatus


def get_model_config() -> dict:
    """Read model config from .llm_config.json."""
    config_path = Path(__file__).parent.parent / ".llm_config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            if "model_id" not in config:
                raise ValueError("Missing 'model_id' in config.")
            return config
    except Exception as e:
        raise FileNotFoundError(f"Failed to read model config: {e}")


class ToolExecutor:
    def __init__(self, use_openai: bool = False):
        self.registry = ToolRegistry()
        self.use_openai = use_openai
        self.model = None
        self.tokenizer = None
        self.is_lfm2 = False
        self.loader = None
        self.device = None

        if self.use_openai:
            print("[Executor] ⚠️ Skipping local model load — using OpenAI backend.")
            return  # 🚫 Skip loading local model

        try:
            config = get_model_config()
            model_id = config["model_id"]
            self.loader = config.get("loader", "auto").lower()
            device = config.get("device", "auto")
            torch_dtype = config.get("torch_dtype", "float16")
            use_safetensors = config.get("use_safetensors", False)

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            dtype = (
                torch.bfloat16
                if model_id == "LiquidAI/LFM2-1.2B"
                else getattr(torch, torch_dtype, torch.float16)
            )

            is_repo_id = "/" in model_id and not Path(model_id).is_absolute()
            model_path = model_id if is_repo_id else str(Path(model_id).resolve())

            print(
                f"[Executor] ✅ Using model: {model_path} ({self.loader}) on {device}"
            )

            is_llama3_8b = "meta-llama/Meta-Llama-3-8B" in model_id
            self.is_lfm2 = model_id == "LiquidAI/LFM2-1.2B"
            offload_params = {}
            if is_llama3_8b:
                offload_params = {
                    "low_cpu_mem_usage": True,
                    "offload_folder": "offload",
                }
                print("[Executor] Detected Llama-3-8B: Enabling CPU offloading.")
            elif self.is_lfm2:
                print(
                    "[Executor] Detected LFM2-1.2B: Using bfloat16 and trust_remote_code."
                )

            if self.loader == "gptq":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, local_files_only=not is_repo_id
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=dtype,
                    trust_remote_code=self.is_lfm2,
                    local_files_only=not is_repo_id,
                    use_safetensors=use_safetensors,
                    revision="main",
                    **offload_params,
                )
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif self.loader == "gguf":
                n_gpu_layers = -1 if device == "cuda" else 0
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=8192,
                    verbose=True,
                )
                self.tokenizer = self.model
            elif self.loader == "safetensors":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, local_files_only=not is_repo_id
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=dtype,
                    trust_remote_code=self.is_lfm2,
                    local_files_only=not is_repo_id,
                    use_safetensors=True,
                    **offload_params,
                )
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError(f"Unsupported loader: {self.loader}")

        except Exception as e:
            print(f"[Executor] ❌ Error loading model: {e}")
            raise

    def execute_plan(
        self, plan: list, prompt: str, dry_run: bool = False, max_new_tokens: int = 1024
    ) -> list:
        results = []
        prev_output = None
        step_outputs = {}  # Track outputs as step_i → result

        if not isinstance(plan, list):
            return [
                {
                    "tool": "[executor]",
                    "status": "error",
                    "message": f"Plan must be a list, got {type(plan).__name__}: {plan}",
                }
            ]

        for i, step in enumerate(plan):
            print(f"[Executor] Step {i}: {step} (type: {type(step).__name__})")

            if not isinstance(step, dict):
                results.append(
                    {
                        "tool": f"[step_{i}]",
                        "status": "error",
                        "message": f"Invalid step type: expected dict, got {type(step).__name__}: {step}",
                    }
                )
                continue

            tool_name = step.get("tool")
            if not tool_name or not isinstance(tool_name, str):
                results.append(
                    {
                        "tool": f"[step_{i}]",
                        "status": "error",
                        "message": f"Missing or invalid 'tool' key in step: {step}",
                    }
                )
                continue

            params = step.get("params", {})
            if not isinstance(params, dict):
                results.append(
                    {
                        "tool": tool_name,
                        "status": "error",
                        "message": f"Invalid 'params' format: expected dict, got {type(params).__name__}",
                    }
                )
                continue

            if tool_name not in self.registry.tools:
                print(
                    f"[Executor] ⚠️ Invalid tool: {tool_name}. Falling back to raw chat completion."
                )
                return self.fallback_raw_completion(prompt, max_new_tokens)

            resolved_params = {}
            for k, v in params.items():
                if isinstance(v, str):
                    if "<prev_output>" in v:
                        if isinstance(prev_output, dict):
                            resolved_val = str(prev_output.get("message", prev_output))
                        else:
                            resolved_val = str(prev_output)
                        v = v.replace("<prev_output>", resolved_val)

                    for ref_idx in range(i):
                        ref_token = f"<step_{ref_idx}>"
                        if ref_token in v and f"step_{ref_idx}" in step_outputs:
                            v = v.replace(
                                ref_token, str(step_outputs[f"step_{ref_idx}"])
                            )

                resolved_params[k] = v

            if dry_run:
                results.append(
                    {
                        "tool": tool_name,
                        "status": "dry_run",
                        "params": resolved_params,
                        "message": "Tool not executed (dry run mode)",
                    }
                )
                continue

            try:
                tool_fn = self.registry.get_function(tool_name)
                output = tool_fn(**resolved_params)

                if not isinstance(output, dict):
                    output = {"result": output, "status": "ok", "message": ""}

                if "status" not in output:
                    output["status"] = "ok"

                print(
                    f"[Executor] ✅ Tool '{tool_name}' succeeded with status: {output['status']}"
                )
                result = {
                    "tool": tool_name,
                    "status": output.get("status", "ok"),
                    "message": output.get("message", ""),
                    "result": output.get("result", output),
                }

                results.append(result)
                prev_output = result["result"]
                step_outputs[f"step_{i}"] = prev_output

            except Exception as e:
                print(
                    f"[Executor] ❌ Tool {tool_name} failed: {e}. Falling back to raw chat completion."
                )
                return self.fallback_raw_completion(prompt, max_new_tokens)

        # Only fall back if *none* of the results have status "ok" or "success"
        if all(r.get("status") not in ("ok", "success") for r in results):
            print(
                "[Executor] No valid tools executed. Falling back to raw chat completion."
            )
            return self.fallback_raw_completion(prompt, max_new_tokens)

        return results

    def fallback_raw_completion(
        self, prompt: str, max_new_tokens: int
    ) -> List[Dict[str, Any]]:
        """Fallback to raw chat completion for non-tool queries, model-specific."""
        print("[Executor] Performing raw chat completion")
        if self.use_openai:
            from agent.llm_openai import generate

            print("[Executor] 🔄 Using OpenAI for raw completion")
            try:
                response = generate(prompt, temperature=0.3)
                return [
                    {
                        "tool": "[openai_raw_completion]",
                        "status": "ok",
                        "message": response,
                        "result": response,
                    }
                ]
            except Exception as e:
                return [
                    {
                        "tool": "[openai_raw_completion]",
                        "status": "error",
                        "message": str(e),
                    }
                ]

        else:
            print("[Executor] 🔄 Using LOCAL model for fallback")

            # Load LLMManager only if it wasn't passed into the constructor
            from agent.llm_manager import LLMManager

            local_llm = LLMManager(use_openai=False)
        try:
            response = local_llm.generate(prompt, temperature=0.3)
            return [
                {
                    "tool": "[local_raw_completion]",
                    "status": "ok",
                    "message": response,
                    "result": response,
                }
            ]
        except Exception as e:
            return [
                {"tool": "[local_raw_completion]", "status": "error", "message": str(e)}
            ]
