"""agent/llm_manager.py"""

import os
from pathlib import Path
from typing import Optional
import logging
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
from gptqmodel import GPTQModel  # or from wherever GPTQModel is defined
from utils.find_root_dir import find_project_root

logger = logging.getLogger(__name__)

find_project_root = find_project_root()

# todo: still working on!!!!!
# def load_gptq_model(model_dir: Path | str, dtype="float16", device="cuda", ):
#     """
#     Manual GPTQ loader using GPTQModel if auto-gptq is unavailable.
#     Expects:
#     - config.json (transformer config)
#     - model.safetensors or model.bin
#     - optional: quantize_config.json
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

#     # Ensure str for paths
#     model_dir = str(model_dir) if isinstance(model_dir, Path) else model_dir

#     # Set config paths manually b/c using gptqmodel instead of auto_gptq
#     config_path = os.path.join(model_dir, "config.json")
#     quant_path = os.path.join(model_dir, "quant_config.json")
#     weights_path = os.path.join(model_dir, "model.safetensors")

#     with open(config_path, "r") as f:
#         config = json.load(f)


def load_gptq_model_gptqmodel(
    model_path: str,
    dtype: str = "float16",
    device: str = "cuda",
    device_map: str = "auto",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    model = GPTQModel.from_quantized(
        model_id_or_path=model_path,
        device_map=device_map,
        device=device,
        backend="auto",
        trust_remote_code=True,
    )

    return tokenizer, model


class LLMManager:
    def __init__(
        self,
        model_path: str | Path,
        quantization: str = "full",
        dtype: str = "float16",
        device_map: str = "auto",
    ):
        self.quantization = quantization
        self.dtype = getattr(torch, dtype)
        self.device_map = device_map

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"[LLMManager] ❌ Model path does not exist: {model_path}"
            )

        print(
            f"[LLMManager] 🔍 Loading model from: {model_path} (quant: {quantization})"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, local_files_only=True
        )

        if quantization == "awq":
            if AutoAWQForCausalLM is None:
                raise ImportError("AutoAWQForCausalLM not available.")
            self.model = AutoAWQForCausalLM.from_pretrained(
                model_path, local_files_only=True
            )

        elif quantization == "gptq":
            if GPTQModel is None:
                raise ImportError("GPTQModel not available.")
            self.tokenizer, self.model = load_gptq_model_gptqmodel(
                model_path, dtype=dtype, device="cuda", device_map=device_map
            )

        else:  # full-precision / bnb / safetensors
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=self.dtype,
                local_files_only=True,
            )

        self.eos = self.tokenizer.eos_token_id
        print("[LLMManager] ✅ Model loaded successfully.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str = None,
    ) -> str:
        full_prompt = (
            f"[SYSTEM] {system_prompt}\n[USER] {prompt}\n[ASSISTANT]"
            if system_prompt
            else prompt
        )

        print(f"[LLMManager] 📥 Full prompt:\n{repr(full_prompt)}\n")

        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(
                self.model.device
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature,
                    pad_token_id=self.eos,
                )

            decoded = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

            if decoded.startswith(full_prompt):
                return decoded[len(full_prompt) :].strip()
            return decoded

        except Exception as e:
            print(f"❌ [LLMManager] Generation failed: {e}")
            raise
