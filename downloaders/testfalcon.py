from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)

print("✅ Model loaded from local Hugging Face cache")
