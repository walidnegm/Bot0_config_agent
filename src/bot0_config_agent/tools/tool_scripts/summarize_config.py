"""bot0_config_agent/tools/tool_scripts/summarize_config.py

Scan a directory for config-like files and summarize top-level keys and
potential secret-looking fields.
"""

from pathlib import Path
import os
import json
import yaml
import logging
from typing import Dict, Any
from bot0_config_agent.agent_models.step_status import StepStatus

logger = logging.getLogger(__name__)

SECRET_KEYWORDS = ["token", "key", "secret", "pass", "auth"]


def is_secret(k: str) -> bool:
    """Heuristic check for secret-like keys."""
    return any(x in k.lower() for x in SECRET_KEYWORDS)


def summarize_config(**kwargs) -> Dict[str, Any]:
    """
    Params (kwargs):
        dir (str, optional): Base directory to scan. Defaults to current working dir.

    Returns:
        dict: Standard envelope (status, message, result)
        result = {
            "configs": [
                {"file": str, "keys": [str], "secrets": [str]} | {"file": str, "error": str},
                ...,
            ]
        }
    """
    base_dir = kwargs.get("dir") or os.getcwd()
    base_path = Path(os.path.expanduser(base_dir)).resolve()

    if not base_path.exists() or not base_path.is_dir():
        return {
            "status": StepStatus.ERROR,
            "message": f"Directory '{base_path}' does not exist or is not a directory.",
            "result": {"configs": []},
        }

    summary = []

    known_filenames = {
        ".env",
        "config.json",
        "config.yaml",
        "config.yml",
        "settings.py",
        "pyproject.toml",
        "requirements.txt",
    }

    def extract_kv_lines(path: Path):
        """Naive KEY=VALUE extractor for dotenv/py/toml-like lines."""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            pairs = {}
            for line in lines:
                s = line.strip()
                if "=" in s and not s.startswith("#"):
                    k, v = s.split("=", 1)
                    pairs[k.strip()] = v.strip()
            return pairs
        except Exception as e:
            return {"error": str(e)}

    logger.info(f"[summarize_config] Scanning for config files in: {base_path}")
    for dirpath, _, files in os.walk(base_path):
        for fname in files:
            if fname not in known_filenames and not any(
                hint in fname.lower() for hint in ("config", "env", "setting")
            ):
                continue

            full_path = Path(dirpath) / fname
            rel_path = os.path.relpath(full_path, start=base_path)

            try:
                if fname.endswith(".json"):
                    try:
                        data = json.loads(full_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        summary.append(
                            {"file": rel_path, "error": f"JSON parse error: {e}"}
                        )
                        continue

                    if isinstance(data, dict):
                        keys = list(data.keys())
                        secrets = [k for k in data if is_secret(k)]
                    else:
                        keys, secrets = ["non-dict JSON"], []
                    summary.append({"file": rel_path, "keys": keys, "secrets": secrets})

                elif fname.endswith((".yaml", ".yml")):
                    try:
                        data = yaml.safe_load(full_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        summary.append(
                            {"file": rel_path, "error": f"YAML parse error: {e}"}
                        )
                        continue

                    if isinstance(data, dict):
                        keys = list(data.keys())
                        secrets = [k for k in data if is_secret(k)]
                    else:
                        keys, secrets = ["non-dict YAML"], []
                    summary.append({"file": rel_path, "keys": keys, "secrets": secrets})

                elif (
                    fname.endswith(".py") or fname == ".env" or fname.endswith(".toml")
                ):
                    kv = extract_kv_lines(full_path)
                    if isinstance(kv, dict) and "error" not in kv:
                        keys = list(kv.keys())
                        secrets = [k for k in kv if is_secret(k)]
                    else:
                        keys, secrets = ["<parse error>"], []
                    summary.append({"file": rel_path, "keys": keys, "secrets": secrets})

                else:
                    # Fallback: treat as KV-ish text
                    kv = extract_kv_lines(full_path)
                    if isinstance(kv, dict) and "error" not in kv:
                        keys = list(kv.keys())
                        secrets = [k for k in kv if is_secret(k)]
                    else:
                        keys, secrets = ["<parse error>"], []
                    summary.append({"file": rel_path, "keys": keys, "secrets": secrets})

            except Exception as e:
                summary.append({"file": rel_path, "error": str(e)})

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Scanned {len(summary)} config-like file(s) under '{base_path}'.",
        "result": {"configs": summary},
    }
