"""
Utils help function (just in case)
"""

from typing import Dict, Optional


def normalize_tool_params(params: dict, param_aliases: Optional[Dict] = None) -> Dict:
    """
    Normalize and sanitize a single tool call's parameter dictionary.

    - Applies key aliasing (e.g., 'filename' or 'filepath' → 'path')
        using param_aliases if provided.
    - Detects placeholder path values such as 'path/to/file', 'your_file',
      or strings containing 'placeholder'.
      If a placeholder is detected in the 'path' parameter, rewrites parameters
      to suggest a keyword-based file search instead.
    - Returns the cleaned parameters dict, or a fallback dict for file search
      if a placeholder is found.

    Args:
        params (dict): Parameters for a tool call.
        param_aliases (dict, optional): Mapping for key aliasing
            (e.g., {"filename": "path"}).
            If None, defaults to an empty mapping.

    Returns:
        dict: Normalized and sanitized parameters, ready for tool execution.
    """
    if param_aliases is None:
        param_aliases = {}

    # Apply aliasing (e.g., 'filename' → 'path')
    normalized_params = {param_aliases.get(k, k): v for k, v in params.items()}

    # Check for placeholder path values
    placeholder_path = normalized_params.get("path", "")
    if any(kw in str(placeholder_path) for kw in ["path/to", "your_", "placeholder"]):
        # Log here if you want, e.g., logger.warning(...)
        return {"keywords": ["python", "py"], "root": "."}

    return normalized_params
