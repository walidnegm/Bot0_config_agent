import os
import logging
from dotenv import load_dotenv
import torch
import yaml
from transformers import AutoTokenizer
from gptqmodel import GPTQModel
import logging_config

logger = logging.getLogger(__name__)

load_dotenv()
hf_token = os.getenv("HUGGING_FACE_TOKEN")
MODELS_YAML = "models.yaml"


def try_load(model_id: str, local_only: bool = True):
    """Try to load GPTQ model from local cache, else from HF hub."""
    print(f"Trying {model_id} (local_only={local_only}) ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=hf_token,
            local_files_only=local_only,
        )
        model = GPTQModel.from_quantized(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            token=hf_token,
            local_files_only=local_only,
        )
        logger.info("✅ Model and tokenizer loaded.")
        return tokenizer, model
    except Exception as e:
        logger.error(f"⚠️ Load failed for {model_id} ({e.__class__.__name__}): {e}")
        return None, None


def test_model(model_id: str, prompt: str):
    assert prompt is not None, "Prompt must not be None"

    # Try local cache first, then hub
    tokenizer, model = try_load(model_id, local_only=True)
    if model is None:
        print("Falling back to download from Hugging Face Hub...")
        tokenizer, model = try_load(model_id, local_only=False)
        if model is None:
            print(f"❌ Failed to load model {model_id}")
            return

    # Prepare input
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nMODEL OUTPUT for {model_id}:\n{answer}")


def main():

    prompt = "Q: Write a Python function that returns the sum of two numbers.\nA:"

    logger.info("Getting model info from models.yaml file...")

    # Load models.yaml
    with open(MODELS_YAML, "r") as f:
        models_config = yaml.safe_load(f)

    # * Models to test
    # model_to_test = "Qwen3-1.7B-Instruct" # Tested and works
    # model_to_test = "Deepseek-Coder-1.3B-Instruct" # tested and works
    # model_to_test = "Gemma-2-2B-it" # tested and works, but very slow with VRAM <= 4GB
    model_to_test = "Llama-3.2-3B-Instruct"

    model = next(
        (m for m in models_config["models"] if m["name"] == model_to_test), None
    )
    if model is None:
        logger.error(f"❌ Model '{model_to_test}' not found in YAML.")
        return

    model_id = model["id"]
    quant = model.get("quantization", "")
    logger.info(f"Model to test: {model_id}")

    # Only test GPTQ models for now
    if "gptq" in quant.lower():
        logger.info(f"\n=== Testing model: {model['name']} ({model_id}) ===")
        test_model(model_id, prompt)
    else:
        logger.info(f"\n--- Skipping {model['name']} (not GPTQ) ---")


if __name__ == "__main__":
    main()
