"""
configs/api_models.py

configurations and help menu for API (cloud) LLMs like OpenAI, Anthrop, Gemini, etc.
"""

# * Anthropic (Claude) models
CLAUDE_OPUS = "claude-3-opus-20240229"
CLAUDE_SONNET_3_5 = "claude-3-5-sonnet-20241022"
CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
CLAUDE_HAIKU = "claude-3-haiku-20240307"

# OpenAI models
GPT_35_TURBO = "gpt-3.5-turbo"
GPT_4 = "gpt-4"
GPT_4_1 = "gpt-4.1"
GPT_4_1_MINI = "gpt-4.1-mini"
GPT_4_1_NANO = "gpt-4.1-nano"
GPT_4O = "gpt-4o"
GPT_O3 = "o3"
GPT_O3_MINI = "o3-mini"
GPT_O3_PRO = "o3-pro"

# Google (Gemini) models
GEMINI_1_5_PRO = "gemini-1.5-pro-latest"
GEMINI_1_5_FLASH = "gemini-1.5-flash-latest"


OPENAI_MODELS = {
    GPT_4O: "Multimodal, but supports pure text (API default)",
    GPT_4_1_MINI: "Smaller, cheaper, strong at text/coding",
    GPT_35_TURBO: "Still supported, default version (4K/16K context)",
}

ANTHROPIC_MODELS = {
    CLAUDE_SONNET_3_5: "Balanced performance and speed",
    CLAUDE_HAIKU: "Fastest, cheapest, lighter tasks",
}

GEMINI_MODELS = {
    GEMINI_1_5_PRO: "Google's latest generation Gemini Pro model with a 1M token context window.",
    GEMINI_1_5_FLASH: "Fastest & most efficient, balanced price/performance, multimodal, up to 1M context",
}

def get_llm_provider(model_name: str) -> str:
    if model_name in OPENAI_MODELS:
        return "openai"
    if model_name in ANTHROPIC_MODELS:
        return "anthropic"
    if model_name in GEMINI_MODELS:
        return "gemini"
    raise ValueError(f"Unknown or unsupported API model: {model_name}")

# ##########################################################################
# ADD THIS MISSING FUNCTION
# ##########################################################################
def validate_api_model_name(model_name: str) -> None:
    """Raises ValueError if model_name is not a known cloud API model."""
    if (
        model_name not in OPENAI_MODELS
        and model_name not in ANTHROPIC_MODELS
        and model_name not in GEMINI_MODELS
    ):
        raise ValueError(f"Unknown or unsupported API model: {model_name}")
