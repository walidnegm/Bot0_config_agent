"""tools/retrieval_tool.py

Scan a directory for config-like files and summarize keys / potential secrets.
"""

import os
import json
import yaml
from pathlib import Path
from agent_models.step_status import StepStatus

SECRET_KEYWORDS = ["token", "key", "secret", "pass", "auth"]


def is_secret(k: str) -> bool:
    """Heuristic check for secret-like keys."""
    return any(x in k.lower() for x in SECRET_KEYWORDS)


def retrieval_tool(**kwargs):
    """
    Scan a directory for config-like files and summarize keys / potential secrets.

    Params (kwargs):
        dir (str, optional): Directory to scan. Defaults to ".".

    Returns:
        dict: Standard envelope (status, message, result)
              result = {"configs": [{"file": str, "keys": [str], "secrets": [str]} | {"file": str, "error": str}, ...]}
    """
    base_dir = Path(kwargs.get("dir") or ".").resolve()

    if not base_dir.exists() or not base_dir.is_dir():
        return {
            "status": StepStatus.ERROR,
            "message": f"Directory '{base_dir}' does not exist or is not a directory.",
            "result": {"configs": []},
        }

    summary = []

    config_like = {
        ".env",
        "config.json",
        "config.yaml",
        "config.yml",
        "settings.py",
        "pyproject.toml",
        "requirements.txt",
    }

    def extract_kv_lines(path: Path):
        """Naive KEY=VALUE extractor for .env/.py/.toml-like lines."""
        try:
            with path.open("r", encoding="utf-8") as f:
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

    for dirpath, _, files in os.walk(base_dir):
        for fname in files:
            fpath = Path(dirpath) / fname
            rel_path = os.path.relpath(fpath, start=base_dir)

            try:
                # Filter to “config-like” by exact names or name contains hint
                if fname in config_like or any(
                    x in fname.lower() for x in ["config", "env", "setting"]
                ):
                    if fname.endswith(".json"):
                        try:
                            data = json.loads(fpath.read_text(encoding="utf-8"))
                        except Exception as e:
                            summary.append(
                                {"file": rel_path, "error": f"JSON parse error: {e}"}
                            )
                            continue

                        if isinstance(data, dict):
                            keys = list(data.keys())
                            secrets = [k for k in data if is_secret(k)]
                        else:
                            keys = ["non-dict JSON"]
                            secrets = []
                        summary.append(
                            {"file": rel_path, "keys": keys, "secrets": secrets}
                        )

                    elif fname.endswith((".yaml", ".yml")):
                        try:
                            data = yaml.safe_load(fpath.read_text(encoding="utf-8"))
                        except Exception as e:
                            summary.append(
                                {"file": rel_path, "error": f"YAML parse error: {e}"}
                            )
                            continue

                        if isinstance(data, dict):
                            keys = list(data.keys())
                            secrets = [k for k in data if is_secret(k)]
                        else:
                            keys = ["non-dict YAML"]
                            secrets = []
                        summary.append(
                            {"file": rel_path, "keys": keys, "secrets": secrets}
                        )

                    elif fname.endswith(".py") or fname == ".env":
                        kv = extract_kv_lines(fpath)
                        if isinstance(kv, dict) and "error" not in kv:
                            keys = list(kv.keys())
                            secrets = [k for k in kv if is_secret(k)]
                        else:
                            keys = ["<parse error>"]
                            secrets = []
                        summary.append(
                            {"file": rel_path, "keys": keys, "secrets": secrets}
                        )

                    elif fname.endswith(".toml"):
                        # naive key=value extraction for toml (good-enough secret scan)
                        lines = extract_kv_lines(fpath)
                        if isinstance(lines, dict) and "error" not in lines:
                            keys = list(lines.keys())
                            secrets = [k for k in lines if is_secret(k)]
                        else:
                            keys = ["non-structured TOML"]
                            secrets = []
                        summary.append(
                            {"file": rel_path, "keys": keys, "secrets": secrets}
                        )

                # else: skip non-config-like names silently
            except Exception as e:
                summary.append({"file": rel_path, "error": str(e)})

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Scanned {len(summary)} config-like files under '{base_dir}'.",
        "result": {"configs": summary},
    }
