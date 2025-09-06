"""
Utils help function (just in case)
"""

from typing import Dict, Mapping, Optional, Iterable

PLACEHOLDER_TOKENS = ("path/to", "your_", "placeholder")


def normalize_tool_params(
    params: Dict,
    param_aliases: Optional[Mapping[str, str]] = None,
    allowed_keys: Optional[Iterable[str]] = None,
) -> Dict:
    """
    Normalize and sanitize parameters for a single tool call.

    Steps performed:
      1. **Alias normalization**
         Applies key aliasing if `param_aliases` is provided.
         Example: {"filename": "path"} turns {"filename": "foo.py"} → {"path": "foo.py"}.

      2. **Placeholder detection**
         Detects non-meaningful placeholder values in "path", such as:
           - "path/to/..."
           - "your_..."
           - strings containing "placeholder"
         If found, returns a fallback dict suggesting a keyword-based file search
         (e.g., {"keywords": ["python", "py"], "root": "."}).

      3. **Return normalized dict**
         Returns the cleaned parameter dict, safe for execution.

    Args:
        params (dict): Raw parameters from the planner or LLM.
        param_aliases (dict, optional): Mapping of alias → canonical name.
            Defaults to {}. Example: {"filename": "path"}.

    Returns:
        dict: Normalized and sanitized parameters ready for tool execution.
              May be replaced by a keyword-search dict if placeholders are detected.
    """

    aliases = param_aliases or {}

    # 1) alias keys
    normalized = {aliases.get(k, k): v for k, v in params.items()}

    # 2) optional whitelist
    if allowed_keys is not None:
        allowed_set = set(allowed_keys)
        normalized = {k: v for k, v in normalized.items() if k in allowed_set}

    # 3) placeholder path fallback
    path = str(normalized.get("path", "")).strip().lower()
    if path and any(tok in path for tok in PLACEHOLDER_TOKENS):
        # Suggest a file-search style param contract instead
        return {"keywords": ["python", "py"], "root": "."}

    return normalized
