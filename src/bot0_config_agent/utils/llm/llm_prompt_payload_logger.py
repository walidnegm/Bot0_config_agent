"""
bot0_config_agent/utils/llm/llm_prompt_payload_logger.py

How to set up .env override:

# ---------------- Logging / Debugging ----------------
LOG_PROMPT_MODE=yaml      # yaml | human | json | raw
LOG_PROMPT_LEVEL=INFO     # DEBUG | INFO | WARNING
"""

from __future__ import annotations
import copy
import os
import json
import logging
import textwrap
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional PyYAML
# ──────────────────────────────────────────────────────────────────────────────
try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # PyYAML not installed
    yaml = None  # type: ignore
    _HAS_YAML = False


# ──────────────────────────────────────────────────────────────────────────────
# YAML helpers (literal block scalars for multiline strings)
# ──────────────────────────────────────────────────────────────────────────────
class _LiteralString(str):
    """Marker type to force YAML literal block scalars (|)."""

    pass


def _literal_str_representer(dumper, data: _LiteralString):  # pragma: no cover
    # style="|" => literal block scalars (preserves real newlines)
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


if _HAS_YAML:

    class _LiteralSafeDumper(yaml.SafeDumper):  # type: ignore[name-defined]
        """Safe dumper that knows how to render _LiteralString as a block scalar."""

        pass

    # Register representer for our marker type
    _LiteralSafeDumper.add_representer(_LiteralString, _literal_str_representer)  # type: ignore[name-defined]


def _literalize_multilines(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _literalize_multilines(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_literalize_multilines(v) for v in obj]
    if (
        isinstance(obj, str) and obj.strip()
    ):  # Force _LiteralString for non-empty strings
        return _LiteralString(obj)
    return obj


def _fmt_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _fmt_yaml(obj: Any) -> str:
    """
    YAML dump that prefers literal blocks for multiline strings;
    falls back to JSON.
    """
    if not _HAS_YAML:
        return _fmt_json(obj)
    payload = _literalize_multilines(obj)
    return yaml.dump(  # type: ignore[call-arg]
        payload,
        Dumper=_LiteralSafeDumper,  # type: ignore[name-defined]
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=10_000,  # avoid folding
    ).rstrip()


# ──────────────────────────────────────────────────────────────────────────────
# Pretty "human" mode
# ──────────────────────────────────────────────────────────────────────────────
def _box(title: str, body: str, width: int = 100) -> str:
    width = max(20, width)
    t = f" {title} "
    line = "─" * max(0, width - 2)
    top = f"┌{line}┐"
    mid = f"│{t}{' ' * max(0, width - 2 - len(t))}│"
    out = [top, mid, f"├{line}┤"]
    for ln in body.splitlines() or [""]:
        wrapped = textwrap.wrap(ln, width=width - 4) or [""]
        for w in wrapped:
            out.append(f"│ {w.ljust(width - 4)} │")
    out.append(f"└{line}┘")
    return "\n".join(out)


def _collapse(text: str, max_lines: int) -> Tuple[str, int]:
    lines = text.splitlines()
    total = len(lines)
    if total <= max_lines:
        return text, total
    head = lines[:max_lines]
    tail_count = total - max_lines
    head.append(f"... ({tail_count} more lines truncated)")
    return "\n".join(head), total


def _fmt_human(
    system_prompt: str, user_prompt: str, *, width: int = 100, max_lines: int = 80
) -> str:
    sys_txt, sys_total = _collapse(system_prompt, max_lines=max_lines // 2)
    usr_txt, usr_total = _collapse(user_prompt, max_lines=max_lines)
    sys_block = _box(f"System Prompt  (lines: {sys_total})", sys_txt, width=width)
    usr_block = _box(f"User Prompt    (lines: {usr_total})", usr_txt, width=width)
    return f"{sys_block}\n{usr_block}"


def _fmt_human_payload(
    payload: Dict[str, Any], *, width: int = 100, max_lines: int = 120
) -> str:
    """Human-mode for generic payloads: render top-level keys as boxed sections."""
    sections: list[str] = []
    for k, v in payload.items():
        if isinstance(v, (dict, list)):
            text = _fmt_yaml(v) if _HAS_YAML else _fmt_json(v)
        else:
            text = str(v or "")
        text, total = _collapse(text, max_lines=max_lines)
        sections.append(_box(f"{k}  (lines: {total})", text, width=width))
    return "\n".join(sections)


# ──────────────────────────────────────────────────────────────────────────────
# Redaction
# ──────────────────────────────────────────────────────────────────────────────
_REDACT_NEEDLES = ("api_key", "authorization", "bearer", "token", "secret", "password")


def _redact_text(s: str) -> str:
    redacted = s
    for needle in _REDACT_NEEDLES:
        redacted = redacted.replace(needle, "[REDACTED]").replace(
            needle.upper(), "[REDACTED]"
        )
    return redacted


def _redact_walk(o: Any) -> Any:
    if isinstance(o, dict):
        return {
            k: (_redact_text(v) if isinstance(v, str) else _redact_walk(v))
            for k, v in o.items()
        }
    if isinstance(o, list):
        return [_redact_walk(v) for v in o]
    if isinstance(o, str):
        return _redact_text(o)
    return o


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_level(
    explicit: Optional[int], fallback_env: str, default_level: int
) -> int:
    if explicit is not None:
        return int(explicit)
    env = os.getenv(fallback_env, "").upper()
    if env and hasattr(logging, env):
        return getattr(logging, env)
    return default_level


def log_prompt_dict(
    logger: logging.Logger,
    *,
    label: str,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,  # "human" | "yaml" | "json" | "raw"
    level: Optional[int] = None,  # default INFO (overridable via env)
    width: int = 100,
    max_lines: int = 120,
    redact: bool = False,
) -> None:
    """
    Pretty-print planner/classifier prompts to logs.

    Modes:
      - mode="human": boxed blocks with wrapping/truncation (best for eyeballing).
      - mode="yaml":  YAML with literal block scalars (|-), falls back to JSON.
      - mode="json":  indented JSON.
      - mode="raw":   plain text blocks (exact newlines, no boxes).

    * Notes:
      - Formatting is lazy (only done if the logger level is enabled).
      - Use env overrides to avoid code changes:
          LOG_PROMPT_MODE = human|yaml|json|raw
          LOG_PROMPT_LEVEL = DEBUG|INFO|...

    Example:
        >>> from utils.prompt_logger import log_prompt_dict
        >>> import logging
        >>> log_prompt_dict(
        ...     logger=logging.getLogger(__name__),
        ...     label="Planner",
        ...     system_prompt="You are a tool-planning assistant...",
        ...     user_prompt="Analyze the user's instruction...\\n=== Tool List ===\\n- list_project_files...",
        ...     mode="human",     # or "yaml"/"json"/"raw"
        ...     level=logging.INFO,
        ... )
    """
    mode = (mode or os.getenv("LOG_PROMPT_MODE") or "human").lower()

    if level is None:
        raise ValueError("log level must be explicitly set")
    if not logger.isEnabledFor(level):
        return

    sys = _LiteralString(system_prompt) or ""
    usr = _LiteralString(user_prompt) or ""

    if redact:
        sys = _redact_text(sys)
        usr = _redact_text(usr)

    if mode == "human":
        text = _fmt_human(sys, usr, width=width, max_lines=max_lines)
        logger.log(level, "[%s] Final Prompt (human):\n%s", label, text)
        return

    if mode == "raw":
        text = f"System Prompt:\n{sys}\n\nUser Prompt:\n{usr}"
        logger.log(level, "[%s] Final Prompt (raw):\n%s", label, text)
        return

    payload: Dict[str, Any] = {"system_prompt": sys, "user_prompt": usr}
    if extras:
        payload.update(extras)

    if mode == "json":
        text = _fmt_json(payload)
        logger.log(level, "[%s] Final Prompt (json):\n%s", label, text)
    else:  # yaml default
        text = _fmt_yaml(payload)
        logger.log(level, "[%s] Final Prompt (yaml):\n%s", label, text)


def log_llm_payload(
    logger: logging.Logger,
    *,
    label: str,
    payload: Dict[str, Any],
    mode: Optional[str] = None,  # "human" | "yaml" | "json" | "raw"
    level: Optional[int] = None,  # default DEBUG (overridable via env)
    width: int = 100,
    max_lines: int = 120,
    redact: bool = False,
) -> None:
    """
    Pretty-print an LLM payload (messages, outputs, params) to logs.

    Behavior
    --------
    - **Lazy**: Returns immediately if `logger` isn't enabled for `level`.
    - **Non-mutating**: Never modifies the input `payload`.
    - **YAML like `log_prompt_dict`**: When `mode="yaml"`, multiline strings are
      rendered using YAML *literal block scalars* (`|`) via `_literalize_multilines`,
      so real line breaks are preserved (no `\\n` noise). If PyYAML isn't installed,
      `_fmt_yaml` falls back to JSON.
    - **Optional redaction**: If `redact=True`, performs a minimal pass that replaces
      likely secrets (e.g., api keys, tokens) with `"[REDACTED]"`. Extend as needed.

    Parameters
    ----------
    logger : logging.Logger
        Destination logger.
    label : str
        Short label to prefix the entry (e.g., "LLMManager", "GPTQ").
    payload : Dict[str, Any]
        The payload to log (e.g., {"messages": [...]}, {"output": "..."}).
    mode : str, default "yaml"
        "yaml" for human-friendly YAML (preferred) or "json" for indented JSON.
    level : int, default logging.DEBUG
        Log level for the message.
    redact : bool, default False
        If True, minimally redact likely secrets in string fields.

    Notes
    -----
    - Use this for inputs/outputs to the model when you want consistent, readable logs.
    - For prompts, `log_prompt_dict` remains the convenience helper.
    - overrides:
        LOG_LLM_MODE  = human|yaml|json|raw
        LOG_LLM_LEVEL = DEBUG|INFO|...
    """
    mode = (mode or os.getenv("LOG_LLM_MODE") or "yaml").lower()
    lvl = _resolve_level(level, "LOG_LLM_LEVEL", logging.DEBUG)

    if not logger.isEnabledFor(lvl):
        return

    data: Dict[str, Any] = copy.deepcopy(payload)
    if redact:
        data = _redact_walk(data)

    if mode == "human":
        text = _fmt_human_payload(data, width=width, max_lines=max_lines)
        logger.log(lvl, "[%s] LLM Payload (human):\n%s", label, text)
        return

    if mode == "raw":
        # Raw: best-effort plain text dump
        parts: list[str] = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                v_str = _fmt_yaml(v) if _HAS_YAML else _fmt_json(v)
            else:
                v_str = str(v)
            parts.append(f"{k}:\n{v_str}")
        text = "\n\n".join(parts)
        logger.log(lvl, "[%s] LLM Payload (raw):\n%s", label, text)
        return

    if mode == "json":
        text = _fmt_json(data)
        logger.log(lvl, "[%s] LLM Payload (json):\n%s", label, text)
    else:  # yaml default
        text = _fmt_yaml(data)
        logger.log(lvl, "[%s] LLM Payload:\n%s", label, text)
