"""utils/get_llm_api_keys.py"""

import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


# API key util functions
def get_openai_api_key() -> str:
    """Retrieves the OpenAI API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Please set it in the .env file.")
        raise EnvironmentError("OpenAI API key not found.")
    return api_key


def get_anthropic_api_key() -> str:
    """Retrieves the Anthropic API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Anthropic API key not found. Please set it in the .env file.")
        raise EnvironmentError("Anthropic API key not found.")
    return api_key
