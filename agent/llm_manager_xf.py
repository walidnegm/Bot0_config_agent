"""
LLMManager: Unified interface to load and run local quantized LLMs (GPTQ, GGUF, AWQ).

Supported loaders:
- GPTQ (via gptqmodel)
- GGUF (via llama.cpp)
- AWQ (via autoawq)

Loads model configuration from `models.yaml`. Example structure:

```yaml
gptq_llama:
  model_path: models/llama2-7b-gptq
  loader: gptq
  torch_dtype: float16
  temperature: 0.7

* Note:
GGUF (General-purpose quantized format for efficient inference of large language models
on devices with limited resources, with suffix .gguf):
A binary format designed for efficient inference of transformer models, particularly large
language models.

GPTQ (General-Purpose Quantization for Transformers): A post-training quantization technique for
large language models, reducing model size and improving inference speed while maintaining accuracy.

AWQ (Activation-aware Weight Quantization): A quantization method that protects salient weights
based on activation distribution, preserving model quality while reducing model size and improving
inference efficiency.

* Changed make:
* - add awq
* - use models.yaml
* - use the gptq load (use gptqmodel lib, not autoawq)
* - separated loader / loaders for each loader type (awq, gguf, gptq)
* - seperated generate / generate for each loader type (awq, gguf, gptq)

Example Usage:
    >>> from agent.llm_manager import LLMManager

        # Load a model named 'gptq_llama' from models.yaml
        llm = LLMManager(model_name="gptq_llama")

        # Call the model to generate a tool-calling JSON
        response = llm.generate("summarize the config files")
        print(response)
"""

import logging
from typing import Optional, Literal, Dict, Any, Union
from pathlib import Path
import re
import json
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM
from llama_cpp import Llama
from gptqmodel import GPTQModel  # Ensure GPTQModel is installed or available
from awq import AutoAWQForCausalLM
from loaders.model_config_models import (
    TransformersLoaderConfig,
    LlamaCppLoaderConfig,
    LoaderConfigEntry,
)
from loaders.load_model_config import load_model_config
from utils.find_root_dir import find_project_root

logger = logging.getLogger(__name__)

try:
    root_dir = find_project_root()
except Exception as e:
    raise FileNotFoundError(
        "❌ Could not determine project root. Make sure one of the expected markers exists \
(e.g., .git, requirements.txt, pyproject.toml, README.md)."
    ) from e

ModelLoaderType = Literal["awq", "gptq", "llama_cpp", "transformers"]


class LLMManager:
    """
    Loads and manages local LLMs (GPTQ, GGUF, or AWQ) for inference, with prompt formatting
    and generation support.
    """

    def __init__(self, use_openai: bool = False, model_name: Optional[str] = None):
        """
        Initialize the LLMManager.

        Args:
            use_openai (bool): Whether to use OpenAI backend instead of local models.
        """
        self.use_openai = use_openai
        self.loader: Optional[ModelLoaderType] = None
        self.tokenizer: Optional[Union[PreTrainedTokenizerBase, Llama]] = None
        self.model: Optional[Any] = None

        if self.use_openai:
            logger.info(
                "[LLMManager] ⚠️ Using OpenAI backend, skipping local model loading."
            )
            return

        model_name = model_name or "default"
        entry = load_model_config(model_name)
        self.loader = entry.loader
        config: TransformersLoaderConfig | LlamaCppLoaderConfig = entry.config

        logger.info(f"[LLMManager] 📦 Initializing model: {model_name} ({self.loader})")

        self._load_model(config)

    def _load_model_with_awq(self, config: TransformersLoaderConfig) -> None:
        """
        Load an AWQ quantized model from a BaseHFModelConfig.

        Args:
            config (BaseHFModelConfig): AWQ config as a Pydantic model.
        """
        loader_kwargs = config.model_dump(exclude={"model_id"})  # exclude positional

        self.model = AutoAWQForCausalLM.from_quantized(config.model_id, **loader_kwargs)

        # Tokenizer loaded separately (not handled by from_quantized)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def _load_model_with_gptq(self, config: TransformersLoaderConfig) -> None:
        """
        Load a GPTQ quantized model.

        Args:
            config (BaseHFModelConfig): GPTQ config as a Pydantic model.
        """
        loader_kwargs = config.model_dump(exclude={"model_id"})  # exclude positional

        self.model = GPTQModel.from_quantized(config.model_id, **loader_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
        assert isinstance(
            tokenizer, PreTrainedTokenizerBase
        )  # Ensure it's not Llama tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def _load_model_with_llama_cpp(self, config: LlamaCppLoaderConfig) -> None:
        """
        Load a GGUF llama.cpp model.

        Args:
            config (GGUFModelConfig): GGUF config Pydantic model.
        """
        self.model = Llama(**config.model_dump())
        self.tokenizer = self.model

    def _load_model_with_transformers(self, config: TransformersLoaderConfig) -> None:
        """
        Load a standard Transformers model from Hugging Face hub or local path.

        Args:
            config (BaseHFModelConfig): Transformers config as a Pydantic model.
        """
        loader_kwargs = config.model_dump(exclude={"model_id"})

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            **loader_kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def _load_model(
        self, config: TransformersLoaderConfig | LlamaCppLoaderConfig
    ) -> None:
        """
        Load the model and tokenizer based on the config.

        Args:
            config: BaseHFModelConfig | GGUFModelConfig): Model configuration
                pydantic model.

        Raises:
            ValueError: If the loader type is unsupported.
        """
        # Set device from config if available (BaseHFModelConfig), else infer
        self.device = getattr(
            config, "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Get the model name for logging
        model_name = (
            getattr(config, "model_id", None)
            or getattr(config, "model_path", None)
            or "unknown"
        )

        logger.info(
            f"[LLMManager] ✅ Using model: {model_name} ({self.loader}) on {self.device}"
        )

        # Dispatch based on loader
        if self.loader == "gptq":
            assert isinstance(config, TransformersLoaderConfig)
            self._load_model_with_gptq(config)
        elif self.loader == "awq":
            assert isinstance(config, TransformersLoaderConfig)
            self._load_model_with_awq(config)
        elif self.loader == "transformers":
            assert isinstance(config, TransformersLoaderConfig)
            self._load_model_with_transformers(config)
        elif self.loader == "llama_cpp":
            assert isinstance(config, LlamaCppLoaderConfig)
            self._load_model_with_llama_cpp(config)
        else:
            raise ValueError(f"Unsupported loader: {self.loader}")

    def _generate_gptq(
        self, prompt: str, system_prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerBase
        )  # Ensure we're using a Hugging Face tokenizer, not llama.cpp

        full_prompt = f"[SYSTEM] {system_prompt}\n[USER] {prompt}\n[ASSISTANT]"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if decoded.startswith(full_prompt):
            return decoded[len(full_prompt) :].strip()
        return decoded

    def _generate_gguf(self, messages: list, max_new_tokens: int) -> str:
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."

        output = self.model.create_chat_completion(
            messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.85,
            stop=["</s>"],
        )
        content = output["choices"][0]["message"]["content"].strip()
        match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        return match.group(0) if match else "[]"

    def _generate_awq(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerBase
        )  # Ensure we're using a Hugging Face tokenizer, not llama.cpp

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature
                > 0.0,  # Only sample stochastically if temp > 0. Else, use greedy decoding
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response using the loaded model.

        Args:
            prompt (str): User prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            system_prompt (Optional[str]): Optional system prompt for instruction tuning.

        Returns:
            str: Generated response.
        """
        if self.use_openai:
            raise RuntimeError("OpenAI mode is active — local generation disabled.")

        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."

        system_prompt = system_prompt or (
            'Return only a valid JSON array of tool calls, like [{"tool": "tool_name", "params": {}}]. '
            "No explanations or extra text."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        logger.info(f"[LLMManager] Messages:\n{json.dumps(messages, indent=2)}\n")

        if self.loader == "gptq":
            return self._generate_gptq(
                prompt, system_prompt, max_new_tokens, temperature
            )
        elif self.loader == "gguf":
            return self._generate_gguf(messages, max_new_tokens)
        elif self.loader == "awq":
            return self._generate_awq(prompt, max_new_tokens, temperature)
        else:
            raise ValueError(f"Unsupported loader type: {self.loader}")
