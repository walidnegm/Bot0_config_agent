"""download_llama3_gptq_hf_default_cache.py"""

import os
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Step 1: Set model ID and auth token
load_dotenv()
model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
hf_token = os.getenv("HUGGING_FACE_TOKEN")


if not hf_token:
    logger.error(
        "HUGGING_FACE_TOKEN environment variable not set. Please set your Hugging Face token."
    )
    raise ValueError(
        "HUGGING_FACE_TOKEN environment variable not set. Please set your Hugging Face token."
    )

# Step 2: Download to default Hugging Face cache
logger.info(f"📦 Downloading {model_id} to Hugging Face default cache...")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)

logger.info("✅ Model download complete.")
