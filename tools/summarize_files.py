"""tools/summarize_files.py

Summarize structure and secret-like keys in config-ish files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml
from agent_models.step_status import StepStatus

SECRET_KEYWORDS = ["token", "key", "secret", "pass", "auth"]


def is_secret(k: str) -> bool:
    return any(x in k.lower() for x in SECRET_KEYWORDS)


KVMap = Dict[str, str]
ErrMap = Dict[str, str]


def extract_kv_lines(path: Path) -> Union[KVMap, ErrMap]:
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        pairs: Dict[str, str] = {}
        for line in lines:
            s = line.strip()
            if "=" in s and not s.startswith("#"):
                k, v = s.split("=", 1)
                pairs[k.strip()] = v.strip()
        return pairs
    except Exception as e:
        return {"error": str(e)}


def summarize_files(**kwargs: Any) -> Dict[str, Any]:
    """
    Summarizes the structure and secret-like keys of given config files.

    Args:
        files (list[str] | str): File paths to summarize (also accepts 'paths' alias).

    Returns:
        dict: Standard envelope:
          {
            "status": StepStatus.SUCCESS | StepStatus.ERROR,
            "message": str,
            "result": {
              "summary": [
                {"file": str, "keys": [str], "secrets": [str]},
                ...
              ],
              "errors"?: [{"file": str, "error": str}, ...]  # present if any errors
            } | None
          }
    """
    files_arg: Optional[Union[str, List[str]]] = kwargs.get("files") or kwargs.get(
        "paths"
    )
    if not files_arg:
        return {
            "status": StepStatus.ERROR,
            "message": "No files provided. Please specify 'files' as a list or string.",
            "result": None,
        }

    # Normalize to list[str]
    if isinstance(files_arg, str):
        files: List[str] = [files_arg]
    elif isinstance(files_arg, list):
        files = [str(p) for p in files_arg]
    else:
        return {
            "status": StepStatus.ERROR,
            "message": "Invalid 'files' type; expected str or list[str].",
            "result": None,
        }

    summary: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for f in files:
        full_path = Path(f)
        rel_path = str(full_path)

        if not full_path.exists() or not full_path.is_file():
            summary.append({"file": rel_path, "keys": ["<not found>"], "secrets": []})
            errors.append({"file": rel_path, "error": "Not found or not a file"})
            continue

        try:
            suffix = full_path.suffix.lower()

            if suffix == ".json":
                text = full_path.read_text(encoding="utf-8")
                data = json.loads(text)
                if isinstance(data, dict):
                    keys = list(data.keys())
                    secrets = [k for k in data if is_secret(k)]
                else:
                    keys, secrets = ["[non-dict JSON]"], []
                summary.append({"file": rel_path, "keys": keys, "secrets": secrets})

            elif suffix in {".yaml", ".yml"}:
                text = full_path.read_text(encoding="utf-8")
                data = yaml.safe_load(text)
                if isinstance(data, dict):
                    keys = list(data.keys())
                    secrets = [k for k in data if is_secret(k)]
                else:
                    keys, secrets = ["[non-dict YAML]"], []
                summary.append({"file": rel_path, "keys": keys, "secrets": secrets})

            elif suffix == ".py" or full_path.name == ".env":
                kv = extract_kv_lines(full_path)
                if isinstance(kv, dict) and "error" not in kv:
                    keys = list(kv.keys())
                    secrets = [k for k in kv if is_secret(k)]
                else:
                    keys, secrets = ["<parse error>"], []
                summary.append({"file": rel_path, "keys": keys, "secrets": secrets})

            else:
                # Fallback: naive key=value parsing for other text configs
                kv = extract_kv_lines(full_path)
                if isinstance(kv, dict) and "error" not in kv:
                    keys = list(kv.keys())
                    secrets = [k for k in kv if is_secret(k)]
                else:
                    keys, secrets = ["<parse error>"], []
                summary.append({"file": rel_path, "keys": keys, "secrets": secrets})

        except Exception as e:
            summary.append({"file": rel_path, "keys": ["<error>"], "secrets": []})
            errors.append({"file": rel_path, "error": str(e)})

    if errors:
        return {
            "status": StepStatus.ERROR,
            "message": f"Summarized {len(summary)} file(s) with {len(errors)} error(s).",
            "result": {"summary": summary, "errors": errors},
        }

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Summarized {len(summary)} file(s).",
        "result": {"summary": summary},
    }
