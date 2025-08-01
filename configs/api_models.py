"""
configs/api_models.py

configurations and help menu for API (cloud) LLMs like OpenAI, Anthrop, Gemini, etc.
"""

# *  Anthropic (Claude) models
CLAUDE_OPUS = "claude-3-opus-20240229"  # Most capable, slowest, highest cost
CLAUDE_SONNET_3_5 = "claude-3-5-sonnet-20241022"  # Balanced performance and speed
CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"  # Latest, best for everyday use
CLAUDE_HAIKU = "claude-3-haiku-20240307"  # Fastest, cheapest, lighter tasks

# OpenAI models
GPT_35_TURBO = "gpt-3.5-turbo"  # Still supported, default version (4K/16K context)
GPT_4 = "gpt-4"  # Slower with priority on quality, more expensive token cost
GPT_4_1 = "gpt-4.1"  # Latest flagship, text-only, long context (up to 1M tokens)
GPT_4_1_MINI = "gpt-4.1-mini"  # Smaller, cheaper, strong at text/coding
GPT_4_1_NANO = "gpt-4.1-nano"  # Even smaller, fastest & cheapest
GPT_4O = "gpt-4o"  # Multimodal, but supports pure text (API default)
GPT_O3 = "o3"  # Reasoning-focused, text-only, best for chain-of-thought
GPT_O3_MINI = "o3-mini"  # Smaller reasoning model, lower latency/cost
GPT_O3_PRO = "o3-pro"  # Highest-fidelity reasoning, text-only (premium/expensive)
# GPT_5 = "gpt-5"   # Upcoming model, expected August 2025

# * Google (Gemini) models
GEMINI_2_5_PRO = "gemini-2.5-pro"  # Most advanced reasoning model, multimodal
# (text, audio, images, video)
GEMINI_2_5_FLASH = "gemini-2.5-flash"  # Fastest, most efficient, balanced price/performance, multimodal
GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"  # Most cost-efficient for high throughput, fastest in 2.5 series
GEMINI_2_0_FLASH = "gemini-2.0-flash"  # Next generation features, speed, and real-time streaming (older Flash)
GEMINI_1_5_PRO = "gemini-1.5-pro"  # Older Pro, still capable for complex reasoning
GEMINI_1_5_FLASH = "gemini-1.5-flash"  # Older Flash, fast and versatile, cost-effective

# * For cli help
OPENAI_MODELS = {
    GPT_35_TURBO: "Legacy model, fast & cheap, basic text/coding",
    GPT_4: "Widespread API access, prioritize on quality, slower and more expensive than newer models",
    GPT_4_1: "Latest flagship, best overall, 1M token context, top coding & reasoning",
    GPT_4_1_MINI: "Smaller & cheaper, great for most tasks, faster than standard",
    GPT_4_1_NANO: "Fastest & cheapest, good for high-volume, lighter tasks",
    GPT_4O: "Multimodal (text+image+audio), fast, strong at text, 128K context",
    GPT_O3: "Best for deep reasoning, chain-of-thought, technical tasks",
    GPT_O3_MINI: "Smaller, lower latency/cost, strong at structured logic",
    GPT_O3_PRO: "Highest-fidelity reasoning, slowest & most expensive, research grade",
    # GPT_5:        "Upcoming, unified reasoning + text, not yet released",
}

ANTHROPIC_MODELS = {
    CLAUDE_OPUS: "Most capable, slowest, highest cost",
    CLAUDE_SONNET_3_5: "Balanced performance and speed",
    CLAUDE_SONNET_4: "Latest, best for everyday use",
    CLAUDE_HAIKU: "Fastest, cheapest, lighter tasks",
}

GEMINI_MODELS = {
    GEMINI_2_5_PRO: "Most advanced reasoning, multimodal (text, audio, images, video), \
up to 1M context",
    GEMINI_2_5_FLASH: "Fastest & most efficient, balanced price/performance, multimodal, \
up to 1M context",
    GEMINI_2_5_FLASH_LITE: "Most cost-efficient, high throughput, and fastest in the 2.5 series. \
Multimodal with 1M context.",
    GEMINI_2_0_FLASH: "Previous generation Flash model. Good for speed and real-time \
streaming with multimodal inputs.",
    GEMINI_1_5_PRO: "Previous generation Pro model. Capable for complex reasoning tasks and \
multimodal inputs, with up to 1M context.",
    GEMINI_1_5_FLASH: "Previous generation Flash model. Fast, efficient, and cost-effective for \
high-volume tasks with multimodal capabilities.",
}


def get_llm_provider(model_name: str) -> str:
    if model_name in OPENAI_MODELS:
        return "openai"
    elif model_name in ANTHROPIC_MODELS:
        return "anthropic"
    elif model_name in GEMINI_MODELS:
        return "gemini"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
