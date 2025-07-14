"""temp testing tool to test the quantized model"""

import os
import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
hf_token = os.getenv("HUGGING_FACE_TOKEN")
prompt = "Q: What is the capital of France?\nA:"


def try_load(local_only=True):
    """Try to load model from local cache, else from HF hub."""
    print(f"Trying local_only={local_only} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=hf_token,
            local_files_only=local_only,
        )
        model = GPTQModel.from_quantized(
            model_id,
            device_map="auto",  # Automatically place model on GPU/CPU
            torch_dtype=torch.float16,
            token=hf_token,
            local_files_only=local_only,
        )
        print("✅ Model and tokenizer loaded.")
        return tokenizer, model
    except Exception as e:
        print(f"⚠️ Load failed ({e.__class__.__name__}): {e}")
        return None, None


def main():
    # Try loading model from local cache first, then Hugging Face Hub
    tokenizer, model = try_load(local_only=True)
    if model is None:
        print("Falling back to download from Hugging Face Hub...")
        tokenizer, model = try_load(local_only=False)
        if model is None:
            raise RuntimeError(
                "❌ Failed to load model from both local cache and HF Hub."
            )

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=32, do_sample=True, temperature=0.7, top_p=0.9
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nMODEL OUTPUT:\n", answer)


if __name__ == "__main__":
    main()
