import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMManager:
    _MODEL_PATH = (
        "/root/projects/Bot0_config_agent/"
        "model/models--meta-llama--Meta-Llama-3-8B-Instruct/"
        "snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    )

    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai

        if self.use_openai:
            print("[LLMManager] ⚠️ Skipping local model load — using OpenAI backend.")
            self.tokenizer = None
            self.model = None
            self.eos = None
            return

        try:
            print("[LLMManager] Loading tokenizer & model …")
            self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_PATH, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                self._MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.eos = self.tokenizer.eos_token_id
            print("[LLMManager] Model loaded successfully.")
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
        """
        Feed a prompt to the model and return the generated response.
        Optional `system_prompt` acts as role conditioning (like OpenAI's system message).
        """
        if self.use_openai:
            raise RuntimeError("LLMManager is in OpenAI mode — local generation is disabled.")

        full_prompt = (
            f"[SYSTEM] {system_prompt}\n[USER] {prompt}\n[ASSISTANT]"
            if system_prompt else prompt
        )

        print(f"[LLMManager] Full prompt:\n{repr(full_prompt)}\n")

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

            print(f"[LLMManager] Full decoded output:\n{repr(full_text)}\n")

            # Attempt to remove prompt from output
            if full_text.startswith(full_prompt):
                generated_text = full_text[len(full_prompt):].strip()
            else:
                print("⚠️ [LLMManager] Warning: Prompt prefix not found in output. Returning full decoded text.")
                generated_text = full_text

            # 🔍 New debug block before returning
            print(f"[LLMManager] 🧪 Generated text before return:\n{repr(generated_text)}\n")

            if not isinstance(generated_text, str):
                raise ValueError(f"Generated output is not a string: {type(generated_text)}")

            return generated_text

        except Exception as e:
            print(f"❌ [LLMManager] Generation failed: {e}")
            raise


def ask_llm(prompt: str, temperature: float = 0.0, role: str = None) -> str:
    system_prompt = {
        "copilot": "You are GitHub Copilot, a helpful developer assistant.",
        "ops": "You are an infrastructure and DevOps expert.",
        "qa": "You are a strict QA tester reviewing behavior and stability.",
        "helper": "You are a friendly and clear technical assistant."
    }.get(role, "You are a helpful assistant.")

    return _llm.generate(
        prompt=prompt,
        temperature=temperature,
        system_prompt=system_prompt
    )

