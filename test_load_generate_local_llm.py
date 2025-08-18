from loaders.load_model_configs import load_model_configs
import json
import logging
from agent.llm_manager import LLMManager
import logging_config

logger = logging.getLogger(__name__)

MODEL_NAMES = [
    "lfm2_1_2b",
    "qwen3_1_7b_instruct_gptq",
    "qwen3_4b_awq",
    "deepseek_coder_1_3b_gptq",
    "phi_3_5_mini_awq",
    "gemma_2_2b_gptq",
    "llama_3_2_3b_gptq",
    "llama_2_7b_chat_gptq",
    "llama3_8b",  # Too large for my PC
    "tinyllama_1_1b_chat_gguf",  # This works
]


def main():
    model_name = MODEL_NAMES[0]
    logger.info(f"Testing Model: {model_name}")

    llm = LLMManager(model_name=model_name)

    prompt = "Q: Write a Python function that returns the sum of two numbers."
    logger.info(f"\n Prompt: {prompt}")

    response = llm.generate(user_prompt=prompt)
    logger.info(f"\nResponse: \n{response}")
    print(f"\nResponse: \n{response}")


if __name__ == "__main__":
    main()
