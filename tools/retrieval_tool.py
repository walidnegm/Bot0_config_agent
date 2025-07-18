import os
import json
import yaml
from pathlib import Path

SECRET_KEYWORDS = ["token", "key", "secret", "pass", "auth"]

def is_secret(k):
    return any(x in k.lower() for x in SECRET_KEYWORDS)

def retrieval_tool(**kwargs):
    summary = []

    config_like = {
        ".env", "config.json", "config.yaml", "config.yml",
        "settings.py", "pyproject.toml", "requirements.txt"
    }

    def extract_kv_lines(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            pairs = {}
            for line in lines:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    pairs[k.strip()] = v.strip()
            return pairs
        except Exception as e:
            return {"error": str(e)}

    for dirpath, _, files in os.walk("."):
        for fname in files:
            fpath = Path(dirpath) / fname
            rel_path = os.path.relpath(fpath)

            if fname in config_like or any(x in fname.lower() for x in ["config", "env", "setting"]):
                print(f"[FOUND] {rel_path}")
                try:
                    if fname.endswith(".json"):
                        data = json.loads(fpath.read_text())
                        summary.append({
                            "file": rel_path,
                            "keys": list(data.keys()) if isinstance(data, dict) else ["non-dict JSON"],
                            "secrets": [k for k in data if is_secret(k)] if isinstance(data, dict) else []
                        })

                    elif fname.endswith((".yaml", ".yml")):
                        data = yaml.safe_load(fpath.read_text())
                        summary.append({
                            "file": rel_path,
                            "keys": list(data.keys()) if isinstance(data, dict) else ["non-dict YAML"],
                            "secrets": [k for k in data if is_secret(k)] if isinstance(data, dict) else []
                        })

                    elif fname.endswith(".py") or fname == ".env":
                        kv = extract_kv_lines(fpath)
                        summary.append({
                            "file": rel_path,
                            "keys": list(kv.keys()) if isinstance(kv, dict) else ["<parse error>"],
                            "secrets": [k for k in kv if is_secret(k)] if isinstance(kv, dict) else []
                        })

                    elif fname.endswith(".toml"):
                        lines = extract_kv_lines(fpath)
                        summary.append({
                            "file": rel_path,
                            "keys": list(lines.keys()) if isinstance(lines, dict) else ["non-structured TOML"],
                            "secrets": [k for k in lines if is_secret(k)] if isinstance(lines, dict) else []
                        })

                    else:
                        print(f"[SKIP: not parseable] {rel_path}")

                except Exception as e:
                    summary.append({
                        "file": rel_path,
                        "error": str(e)
                    })
            else:
                print(f"[SKIP] {rel_path}")

    return {
        "status": "ok",
        "message": f"Scanned {len(summary)} config-like files",
        "configs": summary
    }

