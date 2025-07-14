from huggingface_hub import snapshot_download
from pathlib import Path

# Resolve full path to snapshots dir
target_dir = Path(__file__).resolve().parent / "model/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots"

# Create it if needed
target_dir.mkdir(parents=True, exist_ok=True)

# Download to the snapshots dir
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    local_dir=target_dir,
    local_dir_use_symlinks=False
)

