# agent/llm_manager.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMManager:
    """
    Lightweight wrapper around a local LLaMA-3-8B-Instruct model.

    * Uses AutoTokenizer / AutoModelForCausalLM (model-agnostic, safest).
    * Loads weights from a local snapshot directory.
    * Provides a `.generate(prompt: str) -> str` helper.
    """

    _MODEL_PATH = (
        "/root/projects/Bot0_config_agent/"
        "model/models--meta-llama--Meta-Llama-3-8B-Instruct/"
        "snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    )

    def __init__(self):
        print("[LLMManager] Loading tokenizer & model …")
        self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_PATH, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16  # keep memory low; switch to float32 on CPU if needed
        )
        self.eos = self.tokenizer.eos_token_id

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Feed `prompt` to the model and return the raw completion *after* the prompt.
        Temperature is set to 0.0 for deterministic output; change as desired.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # deterministic
                temperature=0.0,
                pad_token_id=self.eos
            )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return only the part generated *after* the prompt
        return full_text[len(prompt):].strip()

