"""
prompts/prompt_utils.py

Centralized, robust extraction & verification utilities for JSON tool plans
produced by LLMs. All modules (Planner, LLMManager, classifiers) should use
these helpers to avoid duplicated regex logic and inconsistent behavior.

Key features:
- Handles code fences and noisy wrappers
- Supports a sentinel line "FINAL_JSON" to disambiguate the intended array
- Finds balanced top-level JSON arrays (ignoring brackets inside strings)
- Structural validation of plan items: each must be {"tool": str, "params": dict}
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# Low-level helpers
# =============================================================================

_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*|\s*```", re.MULTILINE)

_CHAT_PREFIX_RE = re.compile(r"(?i)(system|user|assistant):.*?(?=\n\n|$)", re.MULTILINE | re.DOTALL)  # Added to strip chat prefixes

def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences to reduce parse failures."""
    return _CODE_FENCE_RE.sub("", text or "").strip()


def strip_chat_prefixes(text: str) -> str:
    """Strip repetitive chat prefixes like 'System:', 'User:' to clean noisy local model outputs."""
    return _CHAT_PREFIX_RE.sub("", text).strip()


def _last_sentinel_index(s: str, sentinel: str = "FINAL_JSON") -> int:
    """
    Find the last occurrence of a LINE that equals the sentinel (case-sensitive),
    return the char index just after that line. -1 if not found.
    """
    matches = list(re.finditer(rf"(?m)^\s*{re.escape(sentinel)}\s*$", s))
    return matches[-1].end() if matches else -1


def _scan_balanced_json_array(s: str, start_pos: int) -> Optional[str]:
    """
    From start_pos, find the first '[' and return the substring of the balanced
    JSON array including nested objects/arrays. Ignores brackets inside quoted strings.
    Returns None if not found or unbalanced.
    """
    n = len(s)
    i = start_pos
    while i < n and s[i] != "[":
        i += 1
    if i >= n:
        return None

    depth = 0
    in_string = False
    escape = False
    start_idx = i

    while i < n:
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return s[start_idx:i+1]
        i += 1
    return None


def extract_last_valid_plan_text(
    raw_text: str,
    use_sentinel: bool = True,
    sentinel: str = "FINAL_JSON",
) -> str:
    """
    Extract the text of the last valid JSON array from raw LLM output.
    Prioritizes the array after the last sentinel, falls back to the last balanced array.
    """
    text = strip_code_fences(strip_chat_prefixes(raw_text)).strip()  # Added strip_chat_prefixes

    if use_sentinel:
        sentinel_idx = _last_sentinel_index(text, sentinel)
        if sentinel_idx >= 0:
            array_text = _scan_balanced_json_array(text, sentinel_idx)
            if array_text:
                return array_text

    # Fallback: find the last balanced array in the text
    pos = 0
    last_array = "[]"
    while True:
        array_text = _scan_balanced_json_array(text, pos)
        if array_text is None:
            break
        last_array = array_text
        pos += len(array_text)

    return last_array


def parse_plan_json(plan_text: str) -> List[Dict[str, Any]]:
    """
    Parse a JSON array string into a Python list. Returns [] on failure
    instead of raising (caller can decide how to fallback).
    """
    try:
        data = json.loads(plan_text)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def validate_plan_structure(
    plan: Sequence[Dict[str, Any]],
    allowed_tools: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Validate that each plan item looks like: {"tool": str, "params": dict}.
    Optionally enforce that the tool name is in `allowed_tools`.

    Returns a filtered list of only valid items.
    """
    allowed = set(allowed_tools) if allowed_tools else None
    validated: List[Dict[str, Any]] = []

    for idx, item in enumerate(plan):
        if not isinstance(item, dict):
            continue
        tool = item.get("tool")
        params = item.get("params")
        if not isinstance(tool, str) or not tool.strip():
            continue
        if not isinstance(params, dict):
            # Some models put null/[] — normalize to empty dict
            if params in (None, []):
                params = {}
            else:
                continue
        if allowed is not None and tool not in allowed:
            # Skip unknown tools if a whitelist is enforced by caller
            continue
        validated.append({"tool": tool.strip(), "params": params})
    return validated


def extract_and_validate_plan(
    raw_text: str,
    allowed_tools: Optional[Iterable[str]] = None,
    use_sentinel: bool = True,
    sentinel: str = "FINAL_JSON",
) -> List[Dict[str, Any]]:
    """
    One-shot utility:
      - Extract the most reliable JSON array text from raw LLM output
      - Parse it
      - Validate structure (and optionally whitelist tools)
    """
    plan_text = extract_last_valid_plan_text(
        raw_text=raw_text, use_sentinel=use_sentinel, sentinel=sentinel
    )
    plan = parse_plan_json(plan_text)
    return validate_plan_structure(plan, allowed_tools=allowed_tools)
