# prompts/load_agent_prompts.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, Undefined

from configs.paths import AGENT_PROMPTS
from tools.workbench.tool_registry import ToolRegistry


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
# ToolRegistry → tools helper
# ───────────────────────────
def _to_dict(obj: Any) -> Dict[str, Any]:
    """Best‑effort convert to dict for Pydantic/objects/dicts."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # Pydantic v2: model_dump
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1: dict
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    # Fallbacks
    try:
        return yaml.safe_load(yaml.safe_dump(obj)) or {}
    except Exception:
        try:
            return vars(obj)
        except Exception:
            return {}


def _extract_parameters(spec_dict: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Return {param: {type: "<type>"}} for the template.
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
            # e.g., meta == "string"  → {"type": "string"}
            simple[name] = {"type": (str(m) if m else "string")}
            continue
        simple[name] = {"type": m.get("type", "string")}
    return simple


def _tools_for_template() -> List[Dict[str, Any]]:
    """
    Build the list your template iterates with:
      [
        {
          "name": ...,
          "description": ...,
          "usage_hint": ...,
          "parameters": { param: {type: "...", ...}, ... }
        },
        ...
      ]
    """
    reg = ToolRegistry()
    tools_list: List[Dict[str, Any]] = []
    for name, spec in getattr(reg, "tools", {}).items():
        sd = _to_dict(spec)
        tools_list.append(
            {
                "name": sd.get("name", name),
                "description": sd.get("description", ""),
                "usage_hint": sd.get("usage_hint", None),
                "parameters": _extract_parameters(sd),
            }
        )
    tools_list.sort(key=lambda d: d["name"])
    return tools_list


# ───────────────────────────
# Render helpers
# ───────────────────────────
def _render_template_to_config(
    template_path: Path | str,
    *,
    jinja_env: Optional[Environment] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Render the Jinja template to YAML and parse to a top-level mapping (dict)."""
    path = Path(template_path)
    env = jinja_env or _JINJA_ENV
    template = env.get_template(path.name)

    # Defensive defaults so template never sees undefined
    kw = dict(kwargs)
    kw.setdefault("user_task", "")
    kw.setdefault("tools", _tools_for_template())

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
    **kwargs: Any,
) -> Dict[str, Any]:
    cfg = _render_template_to_config(template_path, jinja_env=jinja_env, **kwargs)
    if section not in cfg:
        raise KeyError(f"Section '{section}' not found in {template_path}")
    sec = cfg[section] or {}
    if not isinstance(sec, dict):
        raise ValueError(f"Section '{section}' must be a mapping.")
    return sec


# ───────────────────────────
# Public loaders
# ───────────────────────────
def load_planner_prompts(
    *,
    user_task: str,
    template_path: Path | str = AGENT_PROMPTS,
    jinja_env: Optional[Environment] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Loads the 'planner' section, auto‑injecting ToolRegistry metadata.
    Ensures the template sees both 'user_task' and 'tools'.
    """
    return _load_section(
        "planner",
        template_path=template_path,
        jinja_env=jinja_env,
        user_task=user_task,
        tools=_tools_for_template() if tools is None else tools,
    )


def load_intent_classifier_prompts(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, Any]:
    """Loads the full 'intent_classifier' section."""
    return _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
    )


def load_task_decomposition_prompt(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, Any]:
    """Loads just the 'task_decomposition' subkey from 'intent_classifier'."""
    prompts = _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
    )
    task_decomp_prompts = prompts["task_decomposition"]
    if not isinstance(task_decomp_prompts, dict):
        raise TypeError(
            f"Expected a dict for task_decomposition, got {type(task_decomp_prompts).__name__} ({task_decomp_prompts!r})"
        )
    return task_decomp_prompts


def load_describe_only_prompt(
    user_task: str = "",
    template_path: Path | str = AGENT_PROMPTS,
) -> Dict[str, Any]:
    """Loads just the 'describe_only' subkey from 'intent_classifier'."""
    prompts = _load_section(
        section="intent_classifier",
        template_path=template_path,
        user_task=user_task,
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
) -> Dict[str, str]:
    """
    Render the summarizer block; returns {"system_prompt": "...", "prompt": "..."}.
    """
    cfg = _render_template_to_config(
        template_path,
        jinja_env=jinja_env,
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
