from huggingface_hub import snapshot_download
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("🚀 Downloading Falcon RW-1B...")
path = snapshot_download(
    repo_id="tiiuae/falcon-rw-1b",
    local_dir="model/falcon-rw-1b",
    local_dir_use_symlinks=False,
    resume_download=True
)
print(f"✅ Falcon downloaded to: {path}")


