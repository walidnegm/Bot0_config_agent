# utils/get_llm_api_keys.py
from __future__ import annotations

import os
from typing import Dict, Optional

from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


def get_api_keys() -> Dict[str, Optional[str]]:
    """
    Non-throwing aggregator used by utils.llm_api_async.

    Returns:
        dict with keys:
          - 'openai'    -> OPENAI_API_KEY or None
          - 'anthropic' -> ANTHROPIC_API_KEY or None
          - 'google'    -> GOOGLE_API_KEY (Gemini) or None

    We intentionally do NOT raise here. The API caller will log a warning and
    gracefully no-op when a key is missing.
    """
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
    }


# -----------------------------------------------------------------------------
# Strict helpers for call sites that want hard failures
# -----------------------------------------------------------------------------

def get_openai_api_key() -> str:
    """Return OPENAI_API_KEY or raise if missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return api_key


def get_anthropic_api_key() -> str:
    """Return ANTHROPIC_API_KEY or raise if missing."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
    return api_key


def get_google_api_key() -> str:
    """Return GOOGLE_API_KEY (Gemini) or raise if missing."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return api_key

