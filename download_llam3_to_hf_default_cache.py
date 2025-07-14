import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Step 1: Set model ID and auth token
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = os.getenv("HF_TOKEN")

# ✅ Step 2: (No cache_dir needed for default cache)
print(f"📦 Downloading {model_id} to Hugging Face default cache...")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)

print("✅ Model download complete.")
