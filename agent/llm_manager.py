from typing import Optional
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama
import json
import re


def get_model_config_from_config() -> dict:
    config_path = Path(__file__).parent.parent / ".llm_config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            if "model_id" not in config:
                raise ValueError("Missing 'model_id' in config.")
            return config
    except Exception as e:
        raise FileNotFoundError(f"Failed to read model config: {e}")


class LLMManager:
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        self.loader = None  # Store loader type for generate method
        self.device = None  # Store device for generate method

        if self.use_openai:
            print("[LLMManager] ⚠️ Skipping local model load — using OpenAI backend.")
            self.tokenizer = None
            self.model = None
            return

        try:
            print("[LLMManager] 🔍 Locating local model…")

            config = get_model_config_from_config()
            model_id = config["model_id"]
            self.loader = config.get("loader", "auto").lower()
            device = config.get("device", "auto")
            torch_dtype = config.get("torch_dtype", "float16")
            use_safetensors = config.get("use_safetensors", False)

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            dtype = getattr(torch, torch_dtype, torch.float16)

            # Resolve model_id as absolute path if not already absolute
            model_path = Path(model_id)
            if not model_path.is_absolute():
                root_dir = Path(__file__).parent.parent
                model_path = root_dir / model_id
            model_path = str(model_path.resolve())

            print(
                f"[LLMManager] ✅ Using model: {model_path} ({self.loader}) on {device}"
            )

            if self.loader == "gptq":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, local_files_only=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=dtype,
                    trust_remote_code=False,
                    local_files_only=True,
                    safe_serialization=use_safetensors,
                    revision="main",
                )
                # Set pad_token_id to eos_token_id if not set
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif self.loader == "gguf":
                # Set n_gpu_layers to -1 for full GPU offload if CUDA is available, else 0 for CPU
                n_gpu_layers = -1 if device == "cuda" else 0
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=4096,  # Increased context length
                    chat_format="zephyr",  # Match TinyLlama-Chat's format
                    verbose=True,  # Enable detailed logs for CUDA debugging
                )
                self.tokenizer = self.model  # Use Llama's built-in tokenizer
            else:
                raise ValueError(f"Unsupported loader: {self.loader}")

        except Exception as e:
            print(f"[LLMManager] ❌ Error loading model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        if self.use_openai:
            raise RuntimeError(
                "LLMManager is in OpenAI mode — local generation is disabled."
            )

        # Use a concise system prompt for TinyLlama
        system_prompt = (
            system_prompt
            or 'Return only a valid JSON array of tool calls, like [{"tool": "tool_name", "params": {}}]. No explanations or extra text.'
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Log messages for debugging
        print(f"[LLMManager] Messages:\n{json.dumps(messages, indent=2)}\n")

        try:
            if self.loader == "gptq":
                full_prompt = (
                    f"[SYSTEM] {system_prompt}\n[USER] {prompt}\n[ASSISTANT]"
                    if system_prompt
                    else prompt
                )
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(
                    self.device
                )
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0.0,
                        temperature=temperature if temperature > 0.0 else 1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                full_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                ).strip()

                # Extract generated text (remove prompt if present)
                if full_text.startswith(full_prompt):
                    generated_text = full_text[len(full_prompt) :].strip()
                else:
                    print(
                        "⚠️ [LLMManager] Prompt prefix not found. Returning full decoded text."
                    )
                    generated_text = full_text

            elif self.loader == "gguf":
                output = self.model.create_chat_completion(
                    messages,
                    max_tokens=max_new_tokens,
                    temperature=0.0,  # Strict determinism
                    top_p=0.85,  # Tighter sampling
                    stop=["</s>"],  # Model's EOS token
                )
                generated_text = output["choices"][0]["message"]["content"].strip()

                # Strict JSON extraction
                json_match = re.search(r"\[\s*\{.*?\}\s*\]", generated_text, re.DOTALL)
                if json_match:
                    generated_text = json_match.group(0)
                else:
                    print(
                        f"[LLMManager] ⚠️ No JSON array found in output: {generated_text}"
                    )
                    return "[]"  # Fallback to empty array

            else:
                raise ValueError(f"Unsupported loader: {self.loader}")

            print(
                f"[LLMManager] 🧪 Generated text before return:\n{repr(generated_text)}\n"
            )

            if not isinstance(generated_text, str):
                raise ValueError(
                    f"Generated output is not a string: {type(generated_text)}"
                )

            return generated_text

        except Exception as e:
            print(f"❌ [LLMManager] Generation failed: {e}")
            raise
