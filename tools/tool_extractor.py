"""
tools/tool_extractor.py

Centralized extraction and validation utilities for turning raw LLM text
into a structured plan (list[dict]) of tool calls.

Responsibilities:
- Strip code fences / noise
- Detect optional 'FINAL_JSON' sentinel
- Balanced top-level JSON array scan (ignores brackets inside strings)
- Parse JSON into Python
- Lightweight schema check: [{"tool": <str>, "params": <dict>}, ...]
- Normalization and graceful fallback helpers

This module DOES NOT know about your ToolRegistry. It only ensures the
data structure is sound. The planner decides whether tool names are valid.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Public API
# -------------------------

def extract_plan(raw_text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Attempt to extract a plan (list of tool-call dicts) from raw LLM text.
    Returns None if nothing valid found.
    """
    if not raw_text:
        return None

    # 1) Prefer sentinel-guided extraction
    sentinel_pos = _last_sentinel_index(raw_text)
    if sentinel_pos != -1:
        after = raw_text[sentinel_pos:].lstrip()
        arr_text = _scan_balanced_json_array(after, 0)
        if arr_text:
            plan = _loads_if_valid_list(arr_text)
            if plan is not None and _looks_like_tool_plan(plan):
                return plan

    # 2) Remove markdown fences and try scanning for arrays
    stripped = _strip_code_fences(raw_text.strip())
    arrays = _find_all_top_level_arrays(stripped)
    for arr in reversed(arrays):  # prefer the last array
        plan = _loads_if_valid_list(arr)
        if plan is not None and _looks_like_tool_plan(plan):
            return plan

    # 3) Try a naive last-JSON-object (rarely needed, but harmless)
    obj = _extract_last_json_object_like(stripped)
    if isinstance(obj, list) and _looks_like_tool_plan(obj):
        return obj

    return None


def extract_plan_or_fallback(raw_text: str, fallback_prompt: str) -> List[Dict[str, Any]]:
    """
    Extract a plan; if extraction fails, return a single-step fallback plan
    that calls llm_response_async.
    """
    plan = extract_plan(raw_text)
    if plan is not None:
        return plan
    return [{"tool": "llm_response_async", "params": {"prompt": fallback_prompt}}]


# -------------------------
# Internal helpers
# -------------------------

_SENTINEL = "FINAL_JSON"

_CODE_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*([\s\S]*?)```", re.MULTILINE | re.IGNORECASE
)

def _strip_code_fences(s: str) -> str:
    """
    If the text is fenced, prefer inner content; else remove stray fences.
    """
    m = _CODE_FENCE_RE.findall(s)
    if m:
        # If there are fenced blocks, return the last fenced block (most likely the answer)
        return m[-1].strip()
    # Otherwise, just remove loose fences
    s = re.sub(r"```(?:json|JSON)?\s*", "", s)
    s = s.replace("```", "")
    return s.strip()


def _last_sentinel_index(s: str) -> int:
    """
    Find the last line that equals FINAL_JSON (case-sensitive, ignoring spaces).
    Returns the index AFTER the sentinel line, or -1 if not found.
    """
    matches = list(re.finditer(r"(?m)^\s*FINAL_JSON\s*$", s))
    if not matches:
        return -1
    return matches[-1].end()


def _scan_balanced_json_array(s: str, start_pos: int) -> Optional[str]:
    """
    From start_pos, locate the first '[' and return the substring for a balanced
    JSON array, ignoring brackets inside strings and escaping correctly.
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
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return s[start_idx : i + 1]
        i += 1

    return None


def _find_all_top_level_arrays(s: str) -> List[str]:
    """
    Return all well-formed top-level JSON arrays in s.
    """
    out: List[str] = []
    n = len(s)
    i = 0
    while i < n:
        if s[i] == "[":
            arr = _scan_balanced_json_array(s, i)
            if arr:
                out.append(arr)
                i += len(arr)
                continue
        i += 1
    return out


def _loads_if_valid_list(s: str) -> Optional[List[Any]]:
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, list) else None


def _looks_like_tool_plan(plan: List[Any]) -> bool:
    """
    Very lightweight schema check:
      - list of dicts
      - each item has "tool" (str) and "params" (dict or missing -> {})
    Normalize params to dict.
    """
    if not isinstance(plan, list) or len(plan) == 0:
        return False

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            return False
        tool = step.get("tool")
        params = step.get("params", {})
        if not isinstance(tool, str) or (params is not None and not isinstance(params, dict)):
            return False
    return True


def _extract_last_json_object_like(s: str) -> Any:
    """
    As a last resort, try to find the last {...} or [...] block and parse it.
    This is intentionally simple; the primary paths handle arrays.
    """
    # Try last array first
    arrays = _find_all_top_level_arrays(s)
    if arrays:
        try:
            return json.loads(arrays[-1])
        except Exception:
            pass

    # Then try a naive last object via regex (imperfect but better than nothing)
    m = list(re.finditer(r"\{[\s\S]*\}", s))
    if m:
        raw = m[-1].group(0)
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None

