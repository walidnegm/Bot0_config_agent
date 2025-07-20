import os
from huggingface_hub import snapshot_download

# Make sure this is set in your environment or .env
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise EnvironmentError("Missing HUGGING_FACE_HUB_TOKEN in environment.")

local_dir = "./downloaders/model/tinyllama"

print(f"📦 Downloading model to: {local_dir}")
snapshot_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
    token=hf_token,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("✅ Download complete.")

