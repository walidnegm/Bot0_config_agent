import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMManager:
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai

        if self.use_openai:
            print("[LLMManager] ⚠️ Skipping local model load — using OpenAI backend.")
            self.tokenizer = None
            self.model = None
            self.eos = None
            return

        try:
            print("[LLMManager] 🔍 Locating local LLaMA model…")

            model_root = Path.home() / "projects/Bot0_config_agent/model"
            snapshot_base = model_root / "models--meta-llama--Meta-Llama-3-8B-Instruct" / "snapshots"
            candidates = list(snapshot_base.glob("*"))

            if not candidates:
                raise FileNotFoundError(f"No model snapshot found under: {snapshot_base}")
            
            model_path = candidates[0]
            print(f"[LLMManager] ✅ Using model path: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                local_files_only=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True
            )

            self.eos = self.tokenizer.eos_token_id
            print("[LLMManager] 🚀 Model loaded successfully.")

        except Exception as e:
            print(f"❌ [LLMManager] Failed to load model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str = None
    ) -> str:
        if self.use_openai:
            raise RuntimeError("LLMManager is in OpenAI mode — local generation is disabled.")

        full_prompt = (
            f"[SYSTEM] {system_prompt}\n[USER] {prompt}\n[ASSISTANT]"
            if system_prompt else prompt
        )

        #print(f"[LLMManager] Full prompt:\n{repr(full_prompt)}\n")
        lines = full_prompt.splitlines()
        truncated_prompt = "\n".join(lines[:10])
        print(f"[LLMManager] Full prompt (first 10 lines):\n{truncated_prompt}\n")

        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature,
                    pad_token_id=self.eos
                )

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            lines = full_text.splitlines()
            truncated_text = "\n".join(lines[:10])
            print(f"[LLMManager] Full prompt (first 10 lines):\n{truncated_text}\n")
 
            #print(f"[LLMManager] Full decoded output:\n{repr(full_text)}\n")

            if full_text.startswith(full_prompt):
                generated_text = full_text[len(full_prompt):].strip()
            else:
                #print("⚠️ [LLMManager] Prompt prefix not found. Returning full decoded text.")
                generated_text = full_text

            #print(f"[LLMManager] 🧪 Generated text before return:\n{repr(generated_text)}\n")

            if not isinstance(generated_text, str):
                raise ValueError(f"Generated output is not a string: {type(generated_text)}")

            return generated_text

        except Exception as e:
            print(f"❌ [LLMManager] Generation failed: {e}")
            raise



