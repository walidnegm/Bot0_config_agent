from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)

print("✅ Model and tokenizer loaded successfully.")
