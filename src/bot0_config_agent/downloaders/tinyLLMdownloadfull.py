import os
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# Model IDs
model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"
tokenizer_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Optional local download directory (None = default Hugging Face cache)
local_dir = None  # e.g., "./models/tinyllama_gptq"

# Download tokenizer
print(f"🔽 Downloading tokenizer from: {tokenizer_id}")
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_id,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=False,
    cache_dir=local_dir,
)
print("✅ Tokenizer downloaded and cached.\n")

# Download quantized model
print(f"🔽 Downloading quantized model from: {model_id}")
model = AutoGPTQForCausalLM.from_quantized(
    model_id,
    use_safetensors=True,
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    local_files_only=False,
    cache_dir=local_dir,
)
print("✅ Quantized model downloaded and loaded.\n")

