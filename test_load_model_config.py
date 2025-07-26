from loaders.load_model_config import load_model_config
import json

# model = "Qwen/Qwen3-4B-AWQ"
# model = "kaitchup/Qwen3-1.7B-autoround-4bit-gptq",
# model = "TheBloke/deepseek-coder-1.3b-instruct-GPTQ",
# model = "phi_3_5_mini"

with open("loaders/model_configs.json", "r") as f:
    data = json.load(f)

data_keys = data.keys()

for key in data.keys():
    model = load_model_config(key)
    print(model)
