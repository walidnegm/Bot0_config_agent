from typing import Optional
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
<<<<<<< Updated upstream
from llama_cpp import Llama
=======
from huggingface_hub import snapshot_download, scan_cache_dir
>>>>>>> Stashed changes
import json
import re
from llama_cpp import LlamaGrammar


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


def check_model_in_cache(model_id: str) -> bool:
    """Check if model is in Hugging Face cache, download if missing."""
    # Check if model_id is a Hugging Face repo ID (contains '/')
    is_repo_id = '/' in model_id and not Path(model_id).is_absolute()
    repo_id = model_id if is_repo_id else None

    if is_repo_id:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id and repo.size_on_disk > 0:
                print(f"[LLMManager] ✅ Found {model_id} in cache (size: {repo.size_on_disk / 1e9:.2f} GB)")
                return True
        print(f"[LLMManager] ⚠️ Model {model_id} not found in cache. Downloading...")
        try:
            snapshot_download(repo_id=model_id, allow_patterns=["*.bin", "*.safetensors", "*.json", "*.txt"])
            print(f"[LLMManager] ✅ Downloaded {model_id} to cache")
            return True
        except Exception as e:
            print(f"[LLMManager] ❌ Failed to download {model_id}: {e}")
            return False
    else:
        # Check if local path exists
        if Path(model_id).exists():
            print(f"[LLMManager] ✅ Found local model at {model_id}")
            return True
        print(f"[LLMManager] ❌ Local model path {model_id} does not exist")
        return False


class LLMManager:
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        self.loader = None
        self.device = None
        self.is_lfm2 = False

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

            # Check if model is in cache or exists locally
            if not check_model_in_cache(model_id):
                raise ValueError(f"Cannot proceed: Model {model_id} not found in cache or locally and download failed.")

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            dtype = torch.bfloat16 if model_id == "LiquidAI/LFM2-1.2B" else getattr(torch, torch_dtype, torch.float16)

            # Use model_id directly for repo IDs, resolve path for local models
            is_repo_id = '/' in model_id and not Path(model_id).is_absolute()
            model_path = model_id if is_repo_id else str(Path(model_id).resolve())

            print(
                f"[LLMManager] ✅ Using model: {model_path} ({self.loader}) on {device}"
            )

            # Detect Llama-3-8B for CPU offloading or LFM2-1.2B for specific settings
            is_llama3_8b = "meta-llama/Meta-Llama-3-8B" in model_id
            self.is_lfm2 = model_id == "LiquidAI/LFM2-1.2B"
            offload_params = {}
            if is_llama3_8b:
                offload_params = {"low_cpu_mem_usage": True, "offload_folder": "offload"}
                print("[LLMManager] Detected Llama-3-8B: Enabling CPU offloading.")
            elif self.is_lfm2:
                print("[LLMManager] Detected LFM2-1.2B: Using bfloat16 and trust_remote_code.")

            if self.loader == "gptq":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, local_files_only=not is_repo_id
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=dtype,
<<<<<<< Updated upstream
                    trust_remote_code=False,
                    local_files_only=True,
                    safe_serialization=use_safetensors,
                    revision="main",
=======
                    trust_remote_code=self.is_lfm2,
                    local_files_only=not is_repo_id,
                    use_safetensors=use_safetensors,
                    revision="main",
                    **offload_params
>>>>>>> Stashed changes
                )
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif self.loader == "gguf":
<<<<<<< Updated upstream
                # Set n_gpu_layers to -1 for full GPU offload if CUDA is available, else 0 for CPU
=======
                from llama_cpp import Llama
>>>>>>> Stashed changes
                n_gpu_layers = -1 if device == "cuda" else 0
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
<<<<<<< Updated upstream
                    n_ctx=4096,  # Increased context length
                    chat_format="zephyr",  # Match TinyLlama-Chat's format
                    verbose=True,  # Enable detailed logs for CUDA debugging
=======
                    n_ctx=8192,
                    verbose=True
>>>>>>> Stashed changes
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
                    **offload_params
                )
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError(f"Unsupported loader: {self.loader}")

        except Exception as e:
            print(f"[LLMManager] ❌ Error loading model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        if self.use_openai:
            raise RuntimeError("LLMManager is in OpenAI mode — local generation is disabled.")

<<<<<<< Updated upstream
        # Use a concise system prompt for TinyLlama
        system_prompt = (
            system_prompt
            or 'Return only a valid JSON array of tool calls, like [{"tool": "tool_name", "params": {}}]. No explanations or extra text.'
        )
=======
        system_prompt = system_prompt or "Return only a valid JSON array of tool calls, like [{\"tool\": \"tool_name\", \"params\": {}}]. No explanations or extra text."
>>>>>>> Stashed changes

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        print(f"[LLMManager] Messages:\n{json.dumps(messages, indent=2)}\n")

        try:
<<<<<<< Updated upstream
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
=======
            if self.loader in ["gptq", "safetensors"]:
                if self.is_lfm2:
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        tokenize=True
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            do_sample=True,
                            temperature=0.3,
                            min_p=0.15,
                            repetition_penalty=1.05,
                            max_new_tokens=max_new_tokens
                        )
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False).strip()
>>>>>>> Stashed changes
                else:
                    full_prompt = (
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id>\n{system_prompt}<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id>\n{prompt}<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id>"
                    )
                    inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=temperature > 0.0,
                            temperature=temperature if temperature > 0.0 else 1.0,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                    if full_text.startswith(full_prompt):
                        generated_text = full_text[len(full_prompt):].strip()
                    else:
                        print("⚠️ [LLMManager] Prompt prefix not found. Returning full decoded text.")
                        generated_text = full_text

            elif self.loader == "gguf":
                grammar_text = """
root ::= "[" ws (tool-call ("," ws tool-call)*)? ws "]"
tool-call ::= "{" ws "\"tool\"" ws ":" ws string ws "," ws "\"params\"" ws ":" ws object ws "}"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair ::= string ws ":" ws value
value ::= object | array | string | number | "true" | "false" | "null"
array ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" ( [^"\\] | "\\" . )* "\""
number ::= ("-"? ("0" | [1-9][0-9]*)) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws ::= [ \t\n\r]*
                """
                grammar = LlamaGrammar.from_string(grammar_text)

                output = self.model.create_chat_completion(
                    messages,
                    max_tokens=max_new_tokens,
<<<<<<< Updated upstream
                    temperature=0.0,  # Strict determinism
                    top_p=0.85,  # Tighter sampling
                    stop=["</s>"],  # Model's EOS token
=======
                    temperature=0.0,
                    top_p=0.85,
                    grammar=grammar,
                    stop=["</s>"]
>>>>>>> Stashed changes
                )
                generated_text = output["choices"][0]["message"]["content"].strip()

<<<<<<< Updated upstream
                # Strict JSON extraction
                json_match = re.search(r"\[\s*\{.*?\}\s*\]", generated_text, re.DOTALL)
                if json_match:
                    generated_text = json_match.group(0)
                else:
                    print(
                        f"[LLMManager] ⚠️ No JSON array found in output: {generated_text}"
                    )
                    return "[]"  # Fallback to empty array
=======
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', generated_text, re.DOTALL)
                if json_match:
                    generated_text = json_match.group(0)
                else:
                    print(f"[LLMManager] ⚠️ No JSON array found in output: {generated_text}")
                    return "[]"
>>>>>>> Stashed changes

            else:
                raise ValueError(f"Unsupported loader: {self.loader}")

            print(f"[LLMManager] 🧪 Generated text before return:\n{repr(generated_text)}\n")

            if not isinstance(generated_text, str):
                raise ValueError(f"Generated output is not a string: {type(generated_text)}")

            return generated_text

        except Exception as e:
            print(f"❌ [LLMManager] Generation failed: {e}")
            raise
