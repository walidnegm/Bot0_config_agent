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
from loaders.model_configs_models import (
    TransformersLoaderConfig,
    LlamaCppLoaderConfig,
    AWQLoaderConfig,
    GPTQLoaderConfig,
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
        config: (
            AWQLoaderConfig
            | GPTQLoaderConfig
            | LlamaCppLoaderConfig
            | TransformersLoaderConfig
        ) = entry.config

        logger.info(f"[LLMManager] 📦 Initializing model: {model_name} ({self.loader})")

        self._load_model(config)

    def _load_model_with_awq(self, config: AWQLoaderConfig) -> None:
        """
        Load an AWQ quantized model from a BaseHFModelConfig.

        Args:
            config (BaseHFModelConfig): AWQ config as a Pydantic model.
        """
        loader_kwargs = config.model_dump(
            exclude={"model_id_or_path"}
        )  # exclude positional

        self.model = AutoAWQForCausalLM.from_quantized(
            config.model_id_or_path, **loader_kwargs
        )

        # Tokenizer loaded separately (not handled by from_quantized)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id_or_path, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def _load_model_with_gptq(self, config: GPTQLoaderConfig) -> None:
        """
        Load a GPTQ quantized model.

        Args:
            config (BaseHFModelConfig): GPTQ config as a Pydantic model.
        """
        loader_kwargs = config.model_dump(
            exclude={"model_id_or_path"}
        )  # exclude positional
        loader_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GPTQModel.from_quantized(config.model_id_or_path, **loader_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id_or_path, use_fast=True
        )
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
        # llama-cpp requires keyword function constructure format (not positional)
        loader_kwargs = config.model_dump(exclude={"model_id_or_path"})
        loader_kwargs["model_path"] = config.model_id_or_path
        self.model = Llama(**loader_kwargs)
        self.tokenizer = self.model

    def _load_model_with_transformers(self, config: TransformersLoaderConfig) -> None:
        """
        Load a standard Transformers model from Hugging Face hub or local path.

        Args:
            config (BaseHFModelConfig): Transformers config as a Pydantic model.
        """
        loader_kwargs = config.model_dump(exclude={"model_id_or_path"})

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id_or_path,
            **loader_kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id_or_path, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def _load_model(
        self,
        config: (
            AWQLoaderConfig
            | GPTQLoaderConfig
            | LlamaCppLoaderConfig
            | TransformersLoaderConfig
        ),
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
        model_name = getattr(config, "model_id_or_path", "unknown")

        logger.info(
            f"[LLMManager] ✅ Using model: {model_name} ({self.loader}) on {self.device}"
        )

        # Dispatch based on loader
        if self.loader == "gptq":
            assert isinstance(config, GPTQLoaderConfig)
            self._load_model_with_gptq(config)
        elif self.loader == "awq":
            assert isinstance(config, AWQLoaderConfig)
            self._load_model_with_awq(config)
        elif self.loader == "transformers":
            assert isinstance(config, TransformersLoaderConfig)
            self._load_model_with_transformers(config)
        elif self.loader == "llama_cpp":
            assert isinstance(config, LlamaCppLoaderConfig)
            self._load_model_with_llama_cpp(config)
        else:
            raise ValueError(f"Unsupported loader: {self.loader}")

    def _format_prompt(self, prompt: str, system_prompt: str = "") -> str:
        return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

    def _generate_with_awq(
        self,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        expected_res_type: Literal["json", "text", None] = None,
    ) -> str:
        """
        Generates a response using autoawq library.

        Args:
            prompt (str): User prompt.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature (0 = greedy).
            expected_res_type (Literal["json", "text", None], optional): If "json",
                attempts to extract a JSON array from the output.

        Returns:
            str: Either the raw generated text or a parsed JSON array string.
        """
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerBase
        ), "Expected a Hugging Face tokenizer for AWQ models."

        full_prompt = self._format_prompt(prompt, system_prompt)
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        logger.debug(f"[AWQ] 🔁 Prompt:\n{full_prompt}")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"[AWQ] 🧪 Decoded output:\n{repr(decoded)}")

        if expected_res_type == "json":
            match = re.search(r"\[\s*\{.*?\}\s*\]", decoded, re.DOTALL)
            if match:
                logger.debug("[AWQ] ✅ Extracted JSON block.")
                return match.group(0)
            logger.warning("[AWQ] ⚠️ No valid JSON array found. Returning empty array.")
            return "[]"

        return decoded

    def _generate_with_gptq(
        self,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        expected_res_type: Literal["json", "text", None] = None,
    ) -> str:
        """
        Generate a response from a GPTQ-quantized Hugging Face model.

        Args:
            prompt (str): The user input prompt.
            system_prompt (str): The system-level instruction or behavior guide.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature; 0.0 = deterministic.
            expected_res_type (Literal["json", "text", None], optional):
                If "json", attempts to extract a JSON array from the output.

        Returns:
            str: Generated response, either full text or extracted JSON block.
        """
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerBase
        ), "Expected a Hugging Face tokenizer for GPTQ models."

        full_prompt = self._format_prompt(prompt, system_prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        logger.debug(f"[GPTQ] 🔁 Full prompt:\n{inputs}")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"[GPTQ] 🧪 Raw decoded output:\n{repr(decoded)}")

        if decoded.startswith(full_prompt):
            decoded = decoded[len(full_prompt) :].strip()
            logger.debug(f"[GPTQ] ✂️ Stripped prompt prefix:\n{repr(decoded)}")

        if expected_res_type == "json":
            match = re.search(r"\[\s*\{.*?\}\s*\]", decoded, re.DOTALL)
            if match:
                logger.debug("[GPTQ] ✅ Extracted JSON block.")
                return match.group(0)
            logger.warning("[GPTQ] ⚠️ No valid JSON array found. Returning empty array.")
            return "[]"

        return decoded

    def _generate_with_llama_cpp(
        self,
        messages: list,
        max_new_tokens: int,
        expected_res_type: Literal["json", "text", None] = None,
    ) -> str:
        """
        Generates a response using a GGUF model loaded via llama.cpp.

        Args:
            messages (list): List of message dicts in OpenAI-style chat format.
                * The messages include both prompt and system prompt
                * (llama-cpp will combine them internally)
            max_new_tokens (int): Maximum number of tokens to generate.
            expected_res_type (Literal["json", "text", None], optional): If "json",
                attempts to extract a JSON array from the output.

        Returns:
            str: Either the raw generated text or a parsed JSON array string.
        """
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."

        logger.debug(
            f"[llama_cpp] 🔁 Input messages:\n{json.dumps(messages, indent=2)}"
        )

        output = self.model.create_chat_completion(
            messages,
            max_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.85,
            stop=["</s>"],
        )

        content = output["choices"][0]["message"]["content"].strip()
        logger.debug(f"[llama_cpp] 🧪 Raw output:\n{repr(content)}")

        if expected_res_type == "json":
            match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
            if match:
                logger.debug("[llama_cpp] ✅ Extracted JSON block.")
                return match.group(0)
            logger.warning(
                "[llama_cpp] ⚠️ No valid JSON array found. Returning empty array."
            )
            return "[]"

        return content

    def _generate_with_transformers(
        self,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        expected_res_type: Literal["json", "text", None] = None,
    ) -> str:
        """
        Generate a response using a standard Transformers model.

        Args:
            prompt (str): The user input prompt.
            system_prompt (str): System-level behavior instruction.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            expected_res_type (str | None): Optionally extract JSON from output.

        Returns:
            str: Raw or extracted JSON response.
        """
        assert self.model is not None and self.tokenizer is not None
        assert isinstance(self.tokenizer, PreTrainedTokenizerBase)

        full_prompt = self._format_prompt(prompt, system_prompt)
        logger.debug(f"[Transformers] 🔁 Full prompt:\n{full_prompt}")

        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"[Transformers] 🧪 Decoded output:\n{repr(decoded)}")

        if decoded.startswith(full_prompt):
            decoded = decoded[len(full_prompt) :].strip()
            logger.debug(f"[Transformers] ✂️ Stripped prompt prefix:\n{repr(decoded)}")

        if expected_res_type == "json":
            match = re.search(r"\[\s*\{.*?\}\s*\]", decoded, re.DOTALL)
            if match:
                logger.debug("[Transformers] ✅ Extracted JSON block.")
                return match.group(0)
            logger.warning(
                "[Transformers] ⚠️ No valid JSON array found. Returning empty array."
            )
            return "[]"

        return decoded

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        expected_res_type: Literal["json", "text", None] = None,
    ) -> str:
        """
        Generate a response using the loaded model. Expects output can be a JSON
        array of tool calls.

        Delegates to engine-specific _generate_with_*() method.
        The engine is responsible for handling output formatting (e.g., JSON parsing).
        """
        if self.use_openai:
            raise RuntimeError("OpenAI mode is active — local generation disabled.")

        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."

        # todo: comment out to test the function; still WIP
        # system_prompt = system_prompt or (
        #     "Return a valid JSON array of tool calls. Format: "
        #     '[{ "tool": "tool_name", "params": { ... } }]. '
        #     "The key must be 'tool' (not 'call'), and 'tool' must be one of: "
        #     "summarize_config, llm_response, aggregate_file_content, read_file, "
        #     "seed_parser, make_virtualenv, list_project_files, echo_message, "
        #     "retrieval_tool, locate_file, find_file_by_keyword. "
        #     "Do NOT invent new tool names. For general knowledge or definitions, return []."
        # )
        system_prompt = "You are a helpful assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        logger.info(f"[LLMManager] Messages:\n{json.dumps(messages, indent=2)}\n")
        logger.debug(f"Generating with {self.loader}")

        try:
            if self.loader == "awq":
                response = self._generate_with_awq(
                    prompt,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    expected_res_type,
                )
            elif self.loader == "gptq":
                response = self._generate_with_gptq(
                    prompt,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    expected_res_type,
                )

            elif self.loader == "llama_cpp":
                response = self._generate_with_llama_cpp(
                    messages, max_new_tokens, expected_res_type
                )
            else:
                raise ValueError(f"Unsupported loader type: {self.loader}")

            logger.info(f"[LLMManager] 🧪 Generated text:\n{repr(response)}")

            if not isinstance(response, str):
                raise ValueError(f"Generated output is not a string: {type(response)}")

            # Try to extract valid JSON array if needed
            # json_match = re.search(r"\[\s*\{.*?\}\s*\]", response, re.DOTALL)
            # return json_match.group(0) if json_match else "[]"
            return response

        except Exception as e:
            logger.error(f"❌ [LLMManager] Generation failed: {e}")
            raise
