"""model_benches.py

Quick local-model smoke test for all the models:
load → generate → log → save one consolidated report.

>>> Example Usage:
python -m tests.model_smoke_tests.py --out results/my_run.json --prompt "Explain what VRAM is"

python model_smoke_tests.py --out "outputs_temp/llm_smoke_tests/smoke_test_all_models.json" --prompt "Write a Python function that returns the sum of two numbers."

"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure your logging is configured once at entry point
import logging_config  # noqa: F401

from agent.llm_manager import LLMManager
from agent_models.agent_models import (
    JSONResponse,
    TextResponse,
    CodeResponse,
    ToolCall,
    ToolChain,
)
from utils.prompt_logger import log_llm_payload

logger = logging.getLogger("model_bench")

# ---------------------------------------------------------------------------
# Edit this list if you want to change which models are tested.
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    "lfm2_1_2b",
    "qwen3_1_7b_instruct_gptq",
    "qwen3_4b_awq",
    "deepseek_coder_1_3b_gptq",
    "phi_3_5_mini_awq",
    "gemma_2_2b_gptq",
    "llama_3_2_3b_gptq",
    "llama_2_7b_chat_gptq",
    "llama3_8b",  # likely too large for a small GPU, but included for completeness
    "tinyllama_1_1b_chat_gguf",
]

# Default prompt for a quick “does it generate?” check
DEFAULT_PROMPT = "Q: Write a Python function that returns the sum of two numbers."

# Default output file (overridden by --out or $MODEL_TEST_OUT)
DEFAULT_OUTPUT_FILE = "model_bench_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _response_kind_and_dump(resp: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Normalize the various validated response models to a (kind, dict) pair.
    This prevents type errors where a function expects text but receives a model.
    """
    if isinstance(resp, TextResponse):
        return "TextResponse", resp.model_dump()
    if isinstance(resp, CodeResponse):
        return "CodeResponse", resp.model_dump()
    if isinstance(resp, JSONResponse):
        return "JSONResponse", resp.model_dump()
    if isinstance(resp, ToolCall):
        # Pydantic BaseModel
        return "ToolCall", resp.model_dump()
    if isinstance(resp, ToolChain):
        return "ToolChain", resp.model_dump()
    # Fallback (shouldn't happen if LLMManager always validates)
    try:
        return type(resp).__name__, json.loads(json.dumps(resp))
    except Exception:
        return type(resp).__name__, {"value": repr(resp)}


def _response_preview_text(resp: Any) -> str:
    """
    Produce a short, always-string preview suitable for logs/CLIs.
    (Fixes Pylance/mypy complaints when non-strings are passed to text sinks.)
    """
    if isinstance(resp, TextResponse):
        return resp.content
    if isinstance(resp, CodeResponse):
        return resp.code
    if isinstance(resp, JSONResponse):
        try:
            return json.dumps(resp.data, ensure_ascii=False, indent=2)
        except Exception:
            return str(resp.data)
    if isinstance(resp, ToolCall):
        return json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)
    if isinstance(resp, ToolChain):
        return json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)
    # Unknown → stringify safely
    try:
        return json.dumps(resp, ensure_ascii=False, indent=2)
    except Exception:
        return repr(resp)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main bench runner
# ---------------------------------------------------------------------------
def run_bench(output_path: Path, user_prompt: str) -> Dict[str, Any]:
    run_started = dt.datetime.now(dt.timezone.utc).isoformat()
    results = []

    logger.info("=== Model Bench: %d models ===", len(MODEL_NAMES))
    logger.info("Prompt: %s", user_prompt)
    logger.info("Output file: %s", str(output_path))

    for name in MODEL_NAMES:
        logger.info("\n--- Testing model: %s ---", name)
        rec: Dict[str, Any] = {
            "model_name": name,
            "status": "error",
            "error": None,
            "response_kind": None,
            "preview": None,
            "response_dump": None,
            "timestamps": {"started": dt.datetime.now(dt.timezone.utc).isoformat()},
        }

        llm = None  # sentinel -> ensure that VRAM is cleared even if failed in mid-run
        try:
            llm = LLMManager(model_name=name)

            # generate() already validates into one of the known models
            resp = llm.generate(user_prompt=user_prompt)

            kind, dump = _response_kind_and_dump(resp)
            preview = _response_preview_text(resp)

            # Use prompt_logger’s payload printer to show the RESULT payload
            # (The helper is generic; we just pass a dict of what we want to see.)
            log_llm_payload(
                logger,
                label=f"{name} RESULT",
                payload={
                    "model": name,
                    "kind": kind,
                    "preview": preview,
                    "response": dump,
                },
                mode=os.getenv("LOG_PROMPT_MODE", "yaml"),
                level=logging.INFO,
                redact=False,
            )

            rec.update(
                {
                    "status": "ok",
                    "response_kind": kind,
                    "preview": preview,
                    "response_dump": dump,
                    "timestamps": {
                        **rec["timestamps"],
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                }
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Model %s failed: %s", name, e, exc_info=True)

            # Also emit a structured error “payload” with prompt_logger so it’s easy to spot in logs
            log_llm_payload(
                logger,
                label=f"{name} ERROR",
                payload={
                    "model": name,
                    "error_type": type(e).__name__,
                    "traceback": tb,
                },
                mode=os.getenv("LOG_PROMPT_MODE", "yaml"),
                level=logging.ERROR,
                redact=False,
            )

            rec["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": tb,
            }
            rec["timestamps"]["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()

        finally:
            # Always record the result (even if init failed before llm was set)
            results.append(rec)

            # Then try to free VRAM only if we actually loaded a model
            if llm is not None:
                llm.cleanup_vram_cache()

    run_finished = dt.datetime.now(dt.timezone.utc).isoformat()
    report = {
        "run_meta": {
            "started": run_started,
            "finished": run_finished,
            "prompt": user_prompt,
            "models": MODEL_NAMES,
            "host_env": {
                "MODEL_TEST_OUT": os.getenv("MODEL_TEST_OUT"),
                "LOG_PROMPT_MODE": os.getenv("LOG_PROMPT_MODE", "yaml"),
            },
        },
        "results": results,
    }
    _save_json(output_path, report)
    logger.info("\nSaved report to: %s", str(output_path))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test local models and record outputs."
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.getenv("MODEL_TEST_OUT", DEFAULT_OUTPUT_FILE),
        help="Path to save the consolidated JSON report.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt to send to each model.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).expanduser().resolve()
    run_bench(output_path=out_path, user_prompt=args.prompt)


if __name__ == "__main__":
    main()
