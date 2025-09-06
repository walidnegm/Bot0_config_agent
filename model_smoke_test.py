"""model_bench.py

Simple benchmark harness for a single local LLM loading & inference.

- Uses LLMManager to load a model (from model_configs.yaml).
- Runs one inference with a provided user_prompt.
- Logs system+user prompt with prompt_logger.
- Saves raw + validated response to JSON at output_path.

Example:
    $ python model_bench.py --prompt "Explain KV cache" --model qwen3_4b_awq --out results/qwen_test.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from bot0_config_agent.agent.llm_manager import LLMManager
from bot0_config_agent.utils.llm.llm_prompt_payload_logger import log_prompt_dict

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test local LLM loading")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt text")
    parser.add_argument(
        "--model",
        type=str,
        default="gptq_llama",
        help="Model key in model_configs.yaml",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs_temp/smoke_test_local_llm.json",
        help="Where to write the JSON result",
    )
    return parser.parse_args()


def run_bench(
    *, output_path: Path, user_prompt: str, model_name: Optional[str] = None
) -> None:
    """
    Run one inference and dump the result to a JSON file.

    Args:
        output_path (Path): File to save results into (JSON).
        user_prompt (str): Prompt to send to the model.
        model_name (str, optional): Model key from model_configs.yaml.
    """
    model_name = model_name or "gptq_llama"

    logger.info("[Bench] Loading model: %s", model_name)
    llm = LLMManager(model_name)

    system_prompt = "You are a helpful assistant for smoke testing."

    # Log the prompt
    log_prompt_dict(
        logger=logger,
        label="Bench",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mode="human",
        level=logging.INFO,
    )

    # Generate
    response = llm.generate(
        user_prompt, system_prompt=system_prompt, expected_res_type="text"
    )

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "response": str(response),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("[Bench] ✅ Result written to %s", output_path)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).expanduser().resolve()
    run_bench(output_path=out_path, user_prompt=args.prompt, model_name=args.model)


if __name__ == "__main__":
    main()
