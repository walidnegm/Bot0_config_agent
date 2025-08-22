# agent/llm_openai.py
"""
Simple adapter used by LLMManager for API-backed models.

- If model name starts with "gemini", uses Google Generative AI (google-generativeai).
  Requires: GEMINI_API_KEY environment variable.

- Otherwise, uses OpenAI's Chat Completions API (openai).
  Requires: OPENAI_API_KEY environment variable.

Public surface expected by LLMManager:
  class OpenAIAdapter:
      def __init__(self, model: str): ...
      async def generate_chat_async(self, messages: list[dict], temperature: float = 0.3, **kwargs) -> str: ...
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Collapse role-based chat messages into a single prompt (for Gemini fallback)."""
    system = ""
    user = ""
    assistant = ""
    for m in messages:
        role = (m.get("role") or "").lower()
        if role == "system":
            system = m.get("content", "")
        elif role == "user":
            user = m.get("content", "")
        elif role == "assistant":
            assistant = m.get("content", "")
    prompt = ""
    if system:
        prompt += f"System: {system}\n"
    if user:
        prompt += f"User: {user}\n"
    prompt += "Assistant:"
    if assistant:
        prompt += f" {assistant}"
    return prompt


class OpenAIAdapter:
    def __init__(self, model: str) -> None:
        """
        Args:
            model: API model name. Examples:
                   - OpenAI: "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"
                   - Gemini: "gemini-1.5-flash-latest", "gemini-1.5-pro"
        """
        self.model = model
        self.backend = "gemini" if model.lower().startswith("gemini") else "openai"

        if self.backend == "openai":
            try:
                # OpenAI Python SDK (>=1.0)
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise ImportError(
                    "openai package not installed; `pip install openai`"
                ) from e
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
            self._openai_client = OpenAI(api_key=api_key)
            self._gemini_client = None
            logger.info("[OpenAIAdapter] Using OpenAI backend for model: %s", model)

        else:
            try:
                import google.generativeai as genai  # type: ignore
            except Exception as e:
                raise ImportError(
                    "google-generativeai package not installed; `pip install google-generativeai`"
                ) from e
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
            genai.configure(api_key=api_key)
            self._genai = genai
            self._openai_client = None
            logger.info("[OpenAIAdapter] Using Gemini backend for model: %s", model)

    async def generate_chat_async(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """
        Generate a chat completion and return the text content.

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: sampling temperature
            kwargs: passed through (e.g., max_tokens)
        """
        if self.backend == "openai":
            return await self._generate_openai_async(messages, temperature, **kwargs)
        else:
            return await self._generate_gemini_async(messages, temperature, **kwargs)

    # ---------------- OpenAI path ----------------

    async def _generate_openai_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """
        Uses the OpenAI Chat Completions API.
        """
        assert self._openai_client is not None
        # Normalize knobs
        req = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
        }
        # Support both max_tokens and max_new_tokens naming
        if "max_tokens" in kwargs:
            req["max_tokens"] = int(kwargs["max_tokens"])
        elif "max_new_tokens" in kwargs:
            req["max_tokens"] = int(kwargs["max_new_tokens"])

        def _call():
            resp = self._openai_client.chat.completions.create(**req)  # type: ignore
            choice = resp.choices[0]
            return (choice.message.content or "").strip()

        return await asyncio.to_thread(_call)

    # ---------------- Gemini path ----------------

    async def _generate_gemini_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """
        Uses Google Generative AI (Gemini). We convert the chat to a single prompt string
        for simplicity. If you prefer full multi-turn, switch to `start_chat()`.
        """
        assert self._genai is not None
        prompt = _messages_to_prompt(messages)
        model = self._genai.GenerativeModel(self.model)
        gen_cfg = {
            "temperature": float(temperature),
        }
        if "max_tokens" in kwargs:
            gen_cfg["max_output_tokens"] = int(kwargs["max_tokens"])
        elif "max_new_tokens" in kwargs:
            gen_cfg["max_output_tokens"] = int(kwargs["max_new_tokens"])

        def _call():
            resp = model.generate_content(prompt, generation_config=gen_cfg)  # type: ignore
            # google-generativeai Response has .text convenience property
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            # Fallback: concatenate parts if .text not populated
            try:
                parts = []
                for cand in getattr(resp, "candidates", []) or []:
                    for p in getattr(cand, "content", {}).get("parts", []) or []:
                        t = getattr(p, "text", None)
                        if t:
                            parts.append(t)
                return "\n".join(parts).strip()
            except Exception:
                return ""

        return await asyncio.to_thread(_call)

