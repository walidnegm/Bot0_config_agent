from pathlib import Path
import os
import logging
import json
import yaml

logger = logging.getLogger(__name__)

SECRET_KEYWORDS = ["token", "key", "secret", "pass", "auth"]


def is_secret(k):
    return any(x in k.lower() for x in SECRET_KEYWORDS)


def summarize_config(**kwargs):
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

    def extract_kv_lines(path):
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

    print(f"[DEBUG] Searching for config files in: {os.getcwd()}")

    for dirpath, _, files in os.walk("."):
        for fname in files:
            if fname in known_filenames:
                full_path = Path(dirpath) / fname
                rel_path = os.path.relpath(full_path)

                print(f"[FOUND] {rel_path}")
                try:
                    if fname.endswith(".json"):
                        data = json.loads(full_path.read_text())
                        summary.append(
                            {
                                "file": rel_path,
                                "keys": list(data.keys()),
                                "secrets": [k for k in data if is_secret(k)],
                            }
                        )

                    elif fname.endswith((".yaml", ".yml")):
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
                            summary.append(
                                {"file": rel_path, "keys": ["[non-dict YAML]"]}
                            )

                    elif fname.endswith(".py"):
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
            else:
                print(f"[SKIP] {fname}")

    return {
        "status": "ok",
        "message": f"Scanned {len(summary)} config files",
        "configs": summary,
    }
