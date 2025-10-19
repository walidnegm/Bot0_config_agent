# tools/summarize_config.py
"""
Summarizes config files (JSON, YAML, .env, etc.) in the current project.
MCP-compatible: defines get_tool_definition() and run(params).
"""

import os, json, yaml, logging
from pathlib import Path
from typing import List, Dict, Any
from agent_models.step_status import StepStatus

logger = logging.getLogger(__name__)
SECRET_KEYWORDS = ["token", "key", "secret", "pass", "auth"]

def is_secret(k: str) -> bool:
    return any(x in k.lower() for x in SECRET_KEYWORDS)


def _summarize_configs_internal() -> Dict[str, Any]:
    """Core logic moved here so the tool name 'summarize_config' is not shadowed."""
    summary: List[Dict[str, Any]] = []
    known_filenames = {".env", "config.json", "config.yaml", "config.yml",
                       "settings.py", "pyproject.toml", "requirements.txt"}

    def extract_kv_lines(path: Path) -> Dict[str, str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            pairs = {}
            for line in lines:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    pairs[k.strip()] = v.strip()
            return pairs
        except Exception as e:
            return {"error": str(e)}

    for dirpath, _, files in os.walk("."):
        for fname in files:
            if fname in known_filenames:
                full_path = Path(dirpath) / fname
                rel_path = os.path.relpath(full_path)
                try:
                    if fname.endswith(".json"):
                        data = json.loads(full_path.read_text())
                        summary.append({
                            "file": rel_path,
                            "keys": list(data.keys()),
                            "secrets": [k for k in data if is_secret(k)],
                        })
                    elif fname.endswith((".yaml", ".yml")):
                        data = yaml.safe_load(full_path.read_text())
                        if isinstance(data, dict):
                            summary.append({
                                "file": rel_path,
                                "keys": list(data.keys()),
                                "secrets": [k for k in data if is_secret(k)],
                            })
                        else:
                            summary.append({"file": rel_path, "keys": ["[non-dict YAML]"]})
                    else:
                        lines = extract_kv_lines(full_path)
                        summary.append({
                            "file": rel_path,
                            "keys": list(lines.keys()),
                            "secrets": [k for k in lines if is_secret(k)],
                        })
                except Exception as e:
                    summary.append({"file": rel_path, "error": str(e)})

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Scanned {len(summary)} config files",
        "result": summary,
    }


def get_tool_definition():
    return {
        "name": "summarize_config",
        "description": "Summarizes configuration files (JSON, YAML, .env, etc.) in the project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of config files (ignored if None)."
                }
            },
        },
    }


def run(params: dict | None = None):
    try:
        return _summarize_configs_internal()
    except Exception as e:
        return {
            "status": StepStatus.ERROR,
            "message": f"summarize_config failed: {e}",
            "result": None,
        }

