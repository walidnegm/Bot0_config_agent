from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml
from agent_models.agent_models import StepStatus


SECRET_KEYWORDS = ["token", "key", "secret", "pass", "auth"]


def is_secret(k: str) -> bool:
    return any(x in k.lower() for x in SECRET_KEYWORDS)


def extract_kv_lines(path: Path) -> Union[Dict[str, str], Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        pairs: Dict[str, str] = {}
        for line in lines:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                pairs[k.strip()] = v.strip()
        return pairs
    except Exception as e:
        return {"error": str(e)}


def summarize_files(**kwargs: Any) -> Dict[str, Any]:
    """
    Summarizes the structure and secrets of given config files.

    Args:
        files (list of str): List of file paths to summarize (via 'files' or 'paths' kwarg).

    Returns:
        dict: {
            "status": "success" or "error",
            "summary": list of dicts, each with 'file', 'keys', 'secrets', optionally 'error'
        }
    """
    files: Optional[List[str]] = kwargs.get("files") or kwargs.get("paths")
    if not files:
        return {
            "status": StepStatus.ERROR,
            "message": "No files provided. Please specify files as a list.",
        }

    summary: List[Dict[str, Any]] = []
    for f in files:
        full_path = Path(f)
        rel_path = str(full_path)
        try:
            if full_path.suffix == ".json":
                data = json.loads(full_path.read_text())
                summary.append(
                    {
                        "file": rel_path,
                        "keys": list(data.keys()),
                        "secrets": [k for k in data if is_secret(k)],
                    }
                )
            elif full_path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(full_path.read_text())
                if isinstance(data, dict):
                    summary.append(
                        {
                            "file": rel_path,
                            "keys": list(data.keys()),
                            "secrets": [k for k in data if is_secret(k)],
                        }
                    )
                else:
                    summary.append({"file": rel_path, "keys": ["[non-dict YAML]"]})
            elif full_path.suffix == ".py":
                lines = extract_kv_lines(full_path)
                summary.append(
                    {
                        "file": rel_path,
                        "keys": list(lines.keys()),
                        "secrets": [k for k in lines if is_secret(k)],
                    }
                )
            else:
                kv = extract_kv_lines(full_path)
                summary.append(
                    {
                        "file": rel_path,
                        "keys": (
                            list(kv.keys())
                            if isinstance(kv, dict)
                            else ["<parse error>"]
                        ),
                        "secrets": (
                            [k for k in kv if is_secret(k)]
                            if isinstance(kv, dict)
                            else []
                        ),
                    }
                )
        except Exception as e:
            summary.append({"file": rel_path, "error": str(e)})
    return {
        "status": StepStatus.SUCCESS,
        "summary": summary,
        "message": f"Summarized {len(summary)} file(s).",
    }
