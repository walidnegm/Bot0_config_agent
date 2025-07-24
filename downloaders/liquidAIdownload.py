from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model ID
model_id = "LiquidAI/LFM2-1.2B"

# Download the model and tokenizer to cache
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Model and tokenizer downloaded and cached successfully!")
