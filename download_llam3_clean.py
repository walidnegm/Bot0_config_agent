from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Step 1: Set model ID and auth token
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

hf_token = os.getenv("HF_TOKEN")

# ✅ Step 2: Define local cache path for download
cache_dir = "/root/projects/Bot0_config_agent/model"

# ✅ Step 3: Download and cache model + tokenizer
print(f"📦 Downloading {model_id} to {cache_dir}...")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir)

print("✅ Model download complete.")

