"""
utils/llm_api_async.py
Asynchronous wrappers for calling various cloud LLM APIs with graceful
parameter handling and retries.

This module is intentionally self-contained (no project-internal imports),
so it can be imported very early without causing circular imports.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting per provider, per event loop (prevents "re-used across loops")
# ---------------------------------------------------------------------------

_LIMITERS: Dict[Tuple[str, int], AsyncLimiter] = {}


def _get_rate_limiter(provider: str, rate: int = 5, time_period: float = 1.0) -> AsyncLimiter:
    """
    Get or create an AsyncLimiter for (provider, current_event_loop).
    Avoids re-using the same limiter across different event loops.
    """
    loop = asyncio.get_running_loop()
    key = (provider, id(loop))
    limiter = _LIMITERS.get(key)
    if limiter is None:
        limiter = AsyncLimiter(rate, time_period)
        _LIMITERS[key] = limiter
    return limiter


async def _with_rl_and_retry(coro_factory, provider: str, *, max_attempts: int = 3) -> Any:
    """
    Run an awaitable-producing factory with rate limiting + exponential backoff.
    """
    limiter = _get_rate_limiter(provider)
    attempt = 0
    delay = 0.75
    while True:
        attempt += 1
        async with limiter:
            try:
                return await coro_factory()
            except Exception as e:
                if attempt >= max_attempts:
                    logger.exception("LLM call failed after %d attempts (%s): %s", attempt, provider, e)
                    raise
                await asyncio.sleep(delay + (0.1 * attempt))
                delay *= 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_messages(prompt: Optional[str], messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Accepts either a single `prompt` (string) or full chat `messages` list.
    Returns a well-formed OpenAI/Anthropic-style messages list.
    """
    if messages and isinstance(messages, list):
        return messages
    return [{"role": "user", "content": prompt or ""}]


def _maybe_force_json(text: str, response_format: Optional[str]) -> str:
    """
    If response_format == 'json', try to coerce/validate to JSON text.
    We do not raise on failure; callers can validate later.
    """
    if response_format and response_format.lower() == "json":
        try:
            obj = json.loads(text)
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
    return text


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

async def call_openai_api_async(
    model_id: str,
    *,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    response_format: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Call OpenAI Chat Completions asynchronously. Accepts both `prompt` and `messages`.
    Returns text.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for OpenAI provider.")

    msgs = _coerce_messages(prompt, messages)

    async def _do_call():
        # Prefer the new >=1.0 SDK if available
        try:
            from openai import AsyncOpenAI  # type: ignore
            client = AsyncOpenAI(api_key=api_key)
            resp = await client.chat.completions.create(
                model=model_id,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            return _maybe_force_json(text, response_format)
        except Exception:
            # Fallback to legacy SDK if installed
            try:
                import openai  # type: ignore
                openai.api_key = api_key
                resp = await openai.ChatCompletion.acreate(
                    model=model_id,
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = (resp["choices"][0]["message"]["content"] or "").strip()
                return _maybe_force_json(text, response_format)
            except Exception as e:
                raise

    return await _with_rl_and_retry(_do_call, "openai")


async def call_anthropic_api_async(
    model_id: str,
    *,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    response_format: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Call Anthropic Messages asynchronously. Accepts both `prompt` and `messages`.
    Returns text.
    """
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY for Anthropic provider.")

    msgs = _coerce_messages(prompt, messages)

    async def _do_call():
        from anthropic import AsyncAnthropic  # type: ignore
        client = AsyncAnthropic(api_key=api_key)
        resp = await client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=msgs,
        )
        parts = []
        for block in resp.content:
            text_val = getattr(block, "text", None)
            if text_val:
                parts.append(text_val)
        text = "\n".join(p.strip() for p in parts if p.strip())
        return _maybe_force_json(text, response_format)

    return await _with_rl_and_retry(_do_call, "anthropic")


async def call_gemini_api_async(
    model_id: str,
    *,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    response_format: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Call Google Gemini (Generative AI) asynchronously.
    Supports simple single-turn text via `prompt`. Multi-turn via `messages` is not fully
    implemented here, but we construct a single prompt string if messages are provided.
    """
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY for Google provider.")

    # Build a single prompt string for Gemini
    if messages:
        buf = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            buf.append(f"{role.upper()}: {content}")
        effective_prompt = "\n".join(buf)
    else:
        effective_prompt = prompt or ""

    async def _do_call():
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_id)

        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = await model.generate_content_async(effective_prompt, generation_config=generation_config)
            text = (resp.text or "").strip()
            return _maybe_force_json(text, response_format)
        except AttributeError:
            # Older/alternate async API shim
            loop = asyncio.get_running_loop()
            def _blocking_call():
                return model.generate_content(effective_prompt, generation_config=generation_config)
            resp = await loop.run_in_executor(None, _blocking_call)
            text = (getattr(resp, "text", "") or "").strip()
            return _maybe_force_json(text, response_format)

    return await _with_rl_and_retry(_do_call, "google")


# Convenience provider dispatch ------------------------------------------------

async def call_llm_api_async(
    provider: str,
    model_id: str,
    **kwargs: Any,
) -> str:
    """
    Dispatch to the correct provider by name.
    """
    provider = (provider or "").lower()
    if provider in ("openai", "oai", "oa"):
        return await call_openai_api_async(model_id, **kwargs)
    elif provider in ("anthropic", "claude"):
        return await call_anthropic_api_async(model_id, **kwargs)
    elif provider in ("google", "gemini", "vertex"):
        return await call_gemini_api_async(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

