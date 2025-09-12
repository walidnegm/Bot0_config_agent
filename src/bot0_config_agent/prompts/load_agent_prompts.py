# prompts/load_agent_prompts.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Mapping
from collections import OrderedDict

import yaml
from jinja2 import Environment, FileSystemLoader, Undefined

from bot0_config_agent.configs.paths import AGENT_PROMPTS
from bot0_config_agent.tools.configs.tool_registry import ToolRegistry


logger = logging.getLogger(__name__)

# ───────────────────────────
# Tunables for token-light mode
# ───────────────────────────
MAX_DESC_CHARS: int = 240  # truncate long descriptions to save tokens
DEFAULT_TOOL_MODE: Literal["light", "full"] = "light"


# ───────────────────────────
# Jinja filter: fromyaml
# ───────────────────────────
def fromyaml(text: Any) -> Any:
    """Safe-load a YAML string (pass through dict/list). Returns {} on None/blank."""
    if text is None:
        return {}
    if isinstance(text, (dict, list)):
        return text
    s = str(text)
    if not s.strip():
        return {}
    return yaml.safe_load(s)


# ───────────────────────────
# Global Jinja Environment
# ───────────────────────────
def _make_env(template_path: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        undefined=Undefined,  # lenient; switch to StrictUndefined if desired
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    env.filters["fromyaml"] = fromyaml
    return env


_JINJA_ENV: Environment = _make_env(AGENT_PROMPTS)


# ───────────────────────────
# Generic dict conversion
# ───────────────────────────
def _to_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort convert to dict for Pydantic/objects/dicts."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    try:
        return yaml.safe_load(yaml.safe_dump(obj)) or {}
    except Exception:
        try:
            return vars(obj)
        except Exception:
            return {}


# ───────────────────────────
# Token-light helpers
# ───────────────────────────
def _trim(txt: Optional[str], max_len: int = MAX_DESC_CHARS) -> str:
    s = (txt or "").strip()
    if max_len and len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s


def _extract_parameters(spec_dict: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    (Full mode) Return {param: {type: "<type>"}} for the template.
    Robust to string metas (e.g., "string") and object schemas.
    """
    params = spec_dict.get("parameters") or {}
    params = _to_dict(params)

    props = params.get("properties") or {}
    props = _to_dict(props)

    simple: Dict[str, Dict[str, str]] = {}
    for name, meta in props.items():
        m = _to_dict(meta)
        if not isinstance(m, dict):
            simple[name] = {"type": (str(m) if m else "string")}
            continue
        simple[name] = {"type": m.get("type", "string")}
    return simple


def _min_tool_item(sd: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
    """
    Minimal tool payload for prompts: {name, description}
    """
    return {
        "name": sd.get("name", fallback_name),
        "description": _trim(sd.get("description", "")),
    }


def _full_tool_item(sd: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
    """
    Full tool payload (legacy): {name, description, usage_hint, parameters}
    """
    return {
        "name": sd.get("name", fallback_name),
        "description": _trim(sd.get("description", "")),
        "usage_hint": _trim(sd.get("usage_hint", None)),
        "parameters": _extract_parameters(sd),
    }


def _tools_for_template(
    *,
    mode: Literal["light", "full"] = DEFAULT_TOOL_MODE,
) -> List[Dict[str, Any]]:
    """
    Build the list your template iterates with.

    light → [{"name", "description"}, ...]   (DEFAULT; token-light)
    full  → [{"name", "description", "usage_hint", "parameters"}, ...]
    """
    reg = ToolRegistry()
    tools_list: List[Dict[str, Any]] = []
    for name, spec in getattr(reg, "tools", {}).items():
        sd = _to_dict(spec)
        if mode == "full":
            tools_list.append(_full_tool_item(sd, name))
        else:
            tools_list.append(_min_tool_item(sd, name))
    tools_list.sort(key=lambda d: d["name"])
    return tools_list


# ───────────────────────────
# Render helpers
# ───────────────────────────
def _render_template_to_config(
    template_path: Path | str,
    *,
    jinja_env: Optional[Environment] = None,
    tool_mode: Literal["light", "full"] = DEFAULT_TOOL_MODE,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Render the Jinja template to YAML and parse to a top-level mapping (dict)."""
    path = Path(template_path)
    env = jinja_env or _JINJA_ENV
    template = env.get_template(path.name)

    # Defensive defaults so template never sees undefined
    kw = dict(kwargs)
    kw.setdefault("user_task", "")
    # Only inject minimal tool data by default to save tokens
    kw.setdefault("tools", _tools_map_for_template(mode=tool_mode))

    rendered_yaml: str = template.render(**kw)
    cfg = yaml.safe_load(rendered_yaml) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Rendered prompts must be a YAML mapping at top level.")
    return cfg


def _load_section(
    section: str,
    *,
    template_path: Path | str = AGENT_PROMPTS,
    jinja_env: Optional[Environment] = None,
    tool_mode: Literal["light", "full"] = DEFAULT_TOOL_MODE,
    **kwargs: Any,
) -> Dict[str, Any]:
    cfg = _render_template_to_config(
        template_path, jinja_env=jinja_env, tool_mode=tool_mode, **kwargs
    )
    if section not in cfg:
        raise KeyError(f"Section '{section}' not found in {template_path}")
    sec = cfg[section] or {}
    if not isinstance(sec, dict):
        raise ValueError(f"Section '{section}' must be a mapping.")
    return sec


def _tools_map_for_template(
    *,
    mode: Literal["light", "full"] = DEFAULT_TOOL_MODE,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a name→payload mapping your template can iterate with .items().

    light → {name: {"name", "description"}, ...}
    full  → {name: {"name", "description", "usage_hint", "parameters"}, ...}
    """
    reg = ToolRegistry()
    by_name: list[tuple[str, Dict[str, Any]]] = []
    for name, spec in getattr(reg, "tools", {}).items():
        sd = _to_dict(spec)
        if mode == "full":
            by_name.append((name, _full_tool_item(sd, name)))
        else:
            by_name.append((name, _min_tool_item(sd, name)))

    # deterministic order by name
    by_name.sort(key=lambda kv: kv[0])
    return OrderedDict(by_name)  # OrderedDict is also a Mapping


# ───────────────────────────
# Public loaders
# ───────────────────────────
def load_planner_prompts(
    *,
    user_task: str,
    template_path: Path | str = AGENT_PROMPTS,
    jinja_env: Optional[Environment] = None,
    tools: Optional[Mapping[str, Dict[str, Any]]] = None,
    tool_mode: Literal["light", "full"] = DEFAULT_TOOL_MODE,
) -> Dict[str, Any]:
    """
    Loads the 'planner' section, auto-injecting ToolRegistry metadata.
    Ensures the template sees both 'user_task' and 'tools'.

    Defaults to token-light tools (name + trimmed description).
    """
    return _load_section(
        "planner",
        template_path=template_path,
        jinja_env=jinja_env,
        tool_mode=tool_mode,
        user_task=user_task,
        tools=_tools_map_for_template(mode=tool_mode) if tools is None else tools,
    )


def load_intent_classifier_prompts(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
    *,
    tool_mode: Literal["light", "full"] = DEFAULT_TOOL_MODE,
) -> Dict[str, Any]:
    """Loads the full 'intent_classifier' section (with token-light tools by default)."""
    return _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
        tool_mode=tool_mode,
    )


def load_task_decomposition_prompt(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
    jinja_env: Optional[Environment] = None,
) -> Dict[str, Any]:
    """Loads just the 'task_decomposition' subkey from 'intent_classifier'."""
    prompts = _load_section(
        section="intent_classifier",
        template_path=template_path,
        jinja_env=jinja_env,
        user_task=user_task,
    )
    task_decomp_prompts = prompts["task_decomposition"]
    if not isinstance(task_decomp_prompts, dict):
        raise TypeError(
            f"Expected a dict for task_decomposition, got {type(task_decomp_prompts).__name__} ({task_decomp_prompts!r})"
        )

    # Ensure multi-line strings are preserved
    for key, value in task_decomp_prompts.items():
        if isinstance(value, str) and "\n" in value:
            lines = value.rstrip("\n").split("\n")
            task_decomp_prompts[key] = (
                "\n".join(lines) + "\n"
            )  # Ensure consistent newline

            logger.debug(
                f"Preserved prompt {key}: {repr(task_decomp_prompts[key])}"
            )  # todo: debug; delete later

    # todo: debug; delete later
    logger.debug(
        f"Raw task_decomposition prompts after loading from yaml.j2: {repr(task_decomp_prompts)}"
    )
    for key, value in task_decomp_prompts.items():
        logger.debug(f"Prompt {key}: {repr(value)}")

    return task_decomp_prompts


def load_describe_only_prompt(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
    *,
    tool_mode: Literal["light", "full"] = DEFAULT_TOOL_MODE,
) -> Dict[str, Any]:
    """Loads just the 'describe_only' subkey from 'intent_classifier'."""
    prompts = _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
        tool_mode=tool_mode,
    )
    desc_only_prompts = prompts["describe_only"]
    if not isinstance(desc_only_prompts, dict):
        raise TypeError(
            f"Expected a dict for describe_only, got {type(desc_only_prompts).__name__} ({desc_only_prompts!r})"
        )
    return desc_only_prompts


def load_summarizer_prompt(
    *,
    file_type: str = "code",
    style: Optional[str] = None,
    max_words: Optional[int] = 220,
    outline_json: Optional[str] = None,
    excerpts: str = "",
    as_json: bool = False,
    template_path: Path | str = AGENT_PROMPTS,
    jinja_env: Optional[Environment] = None,
    tool_mode: Literal[
        "light", "full"
    ] = DEFAULT_TOOL_MODE,  # available if template uses tools
) -> Dict[str, str]:
    """
    Render the summarizer block; returns {"system_prompt": "...", "prompt": "..."}.
    Token-light by default if the template happens to reference tools.
    """
    cfg = _render_template_to_config(
        template_path,
        jinja_env=jinja_env,
        tool_mode=tool_mode,
        file_type=file_type,
        style=style,
        max_words=max_words,
        outline_json=outline_json,
        excerpts=excerpts,
    )
    summ = cfg.get("summarizer", {}) or {}
    key = "summarize_prompt_json" if as_json else "summarize_prompt_text"
    return {
        "system_prompt": summ.get("system_prompt", ""),
        "prompt": summ.get(key, ""),
    }
