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
from agent_models.llm_response_models import (
    JSONResponse,
    CodeResponse,
    TextResponse,
    ToolSelect,
    ToolSteps,
)
from agent_models.llm_response_validators import (
    validate_response_type,
    validate_tool_selection_or_steps,
)
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
        ) = entry.config  # Model loading config
        self.generation_config: dict = (
            entry.generation_config or {}
        )  # ✅ Generation config

        logger.info(f"[LLMManager] 📦 Initializing model: {model_name} ({self.loader})")

        self._load_model(config)

    def _load_model_with_awq(self, config: AWQLoaderConfig) -> None:
        """
        Load an AWQ quantized model using AutoAWQForCausalLM.from_quantized.

        This method ensures that:
        - The `device` is properly resolved (defaults to 'cuda' if available for AWQ models).
        - The model is initialized with the positional `model_id_or_path` argument
            passed explicitly.
        This is necessary because `from_quantized()` requires `model_id_or_path`
        as a positional parameter, and we avoid including it in `**loader_kwargs`
        to prevent accidental duplication or argument conflicts.

        Args:
            config (AWQLoaderConfig): Configuration object for the AWQ model, including
                                    model path, device, dtype, and other loader-specific options.
        """
        # Prefer CUDA if available for AWQ models
        if getattr(config, "device", None) in (None, "auto"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        if self.device == "cpu":
            logger.warning(
                "[LLMManager] ⚠️ AWQ model will run on CPU — this may be very slow or unsupported."
            )

        loader_kwargs = config.model_dump(exclude={"model_id_or_path", "device"})

        self.model = AutoAWQForCausalLM.from_quantized(
            config.model_id_or_path,
            device=self.device,
            fuse_layers=False,  # Set to false for now due to hardware constraint (do not have flash attention)
            **loader_kwargs,
        )

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
        expected_res_type: Literal["json", "text", None] = None,
        **generation_kwargs,
    ) -> str:
        """
        Generates a response using autoawq library.

        Args:
            prompt (str): User prompt.
            system_prompt (str): System prompt.
            expected_res_type (Literal["json", "text", None], optional): If "json",
                attempts to extract a JSON array from the output.
            **generation_kwargs: All generation params (max_new_tokens, temperature, etc.)

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
        # input_ids = self.tokenizer(full_prompt, return_tensors="pt", padding=True).input_ids.to(
        #     self.device
        # )
        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(
            self.device
        )  # Need to get attention_mask to pass into the generate method
        # attention_mask = inputs["attention_mask"].to(self.device)

        logger.debug(f"[AWQ] 🔁 Prompt:\n{full_prompt}")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,  # add attention mask
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=generation_kwargs.get("temperature", 0.3) > 0.0,
                **generation_kwargs,
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
        expected_res_type: Literal["json", "text", None] = None,
        **generation_kwargs,
    ) -> str:
        """
        Generate a response from a GPTQ-quantized Hugging Face model.

        Args:
            prompt (str): The user input prompt.
            system_prompt (str): The system-level instruction or behavior guide.
            expected_res_type (Literal["json", "text", None], optional):
                If "json", attempts to extract a JSON array from the output.
            **generation_kwargs: All generation params (max_new_tokens, temperature, etc.)
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
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=generation_kwargs.get("temperature", 0.3) > 0.0,
                **generation_kwargs,
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
        expected_res_type: Literal["json", "text", None] = None,
        **generation_kwargs,
    ) -> str:
        """
        Generates a response using a GGUF model loaded via llama.cpp.

        Args:
            messages (list): List of message dicts in OpenAI-style chat format.
                * The messages include both prompt and system prompt
                * (llama-cpp will combine them internally)
            expected_res_type (str | None): Extract JSON block if "json".
            generation_kwargs (dict): Keys like max_new_tokens, temperature, stop, etc.

        Returns:
            str: Raw or JSON-parsed response string.
        """
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."

        logger.debug(
            f"[llama_cpp] 🔁 Input messages:\n{json.dumps(messages, indent=2)}"
        )

        # llama-cpp requires to upack the kwargs and rebuild it
        kwargs = {
            "messages": messages,
            "max_tokens": generation_kwargs.get("max_tokens", 256),
            "temperature": generation_kwargs.get("temperature", 0.2),
            "top_p": generation_kwargs.get("top_p", 0.95),
            "top_k": generation_kwargs.get(
                "top_k", 40
            ),  # llama-cpp does not allow None
            "stop": generation_kwargs.get("stop", ["</s>"]),
        }

        # Safeguard to ensure no None values in kwargs (llama-cpp specific) and
        # top_k is int
        # Safeguard: cast top_k to int if present
        if "top_k" in kwargs:
            kwargs["top_k"] = int(kwargs["top_k"])

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        output = self.model.create_chat_completion(**kwargs)

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
        expected_res_type: Literal["json", "text", None] = None,
        **generation_kwargs,
    ) -> str:
        """
        Generate a response using a standard Transformers model.

        Args:
            prompt (str): The user input prompt.
            system_prompt (str): System-level behavior instruction.
            expected_res_type (str | None): Optionally extract JSON from output.
            **generation_kwargs: All generation params (max_new_tokens, temperature, etc.)

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
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=generation_kwargs.get("temperature", 0.3) > 0.0,
                **generation_kwargs,
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
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        expected_res_type: Literal["json", "text", None] = "text",
    ) -> Union[JSONResponse, TextResponse, CodeResponse, ToolSelect, ToolSteps]:
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

        # Validate response type before returning
        if expected_res_type not in ["json", "text", "code"]:
            raise ValueError(
                f"Invalid expected_res_type '{expected_res_type}'. "
                "Must be one of: 'json', 'text', or 'code'."
            )
        # System prompt
        system_prompt = "You are a helpful assistant."

        # Load generation config from class
        gen_cfg = self.generation_config.copy()

        # ✅ Allow optional overrides
        if max_new_tokens is not None:
            gen_cfg["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_cfg["temperature"] = temperature

        # Special setting for llama-cpp
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        logger.info(f"[LLMManager] Messages:\n{json.dumps(messages, indent=2)}\n")
        logger.debug(f"Generating with {self.loader}")

        try:
            if self.loader == "awq":
                response = self._generate_with_awq(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    expected_res_type=expected_res_type,
                    **gen_cfg,
                )

            elif self.loader == "gptq":
                response = self._generate_with_gptq(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    expected_res_type=expected_res_type,
                    **gen_cfg,
                )

            elif self.loader == "llama_cpp":
                response = self._generate_with_llama_cpp(
                    messages=messages, expected_res_type=expected_res_type, **gen_cfg
                )

            elif self.loader == "transformers":
                response = self._generate_with_transformers(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    expected_res_type=expected_res_type,
                    **gen_cfg,
                )

            else:
                raise ValueError(f"Unsupported loader type: {self.loader}")

            if not response:
                raise ValueError(f"Generated output is empty [].")

            logger.info(f"[LLMManager] 🧪 Generated text (raw):\n{repr(response)}")

            validated_response = validate_response_type(response, expected_res_type)

            # Handle JSON responses that may be ToolSelect or ToolSteps
            if isinstance(validated_response, JSONResponse):
                try:
                    validated_response = validate_tool_selection_or_steps(
                        validated_response.data
                    )
                    return validated_response
                except Exception as e:
                    logger.error(
                        f"Failed to parse JSONResponse.data as ToolSelect/ToolSteps: {e}"
                    )
                    raise

            elif isinstance(validated_response, (TextResponse, CodeResponse)):
                return validated_response

            else:
                logger.error(
                    f"Validated response has unsupported type: {type(validated_response)}; "
                    f"Value: {repr(validated_response)}"
                )
                raise TypeError(
                    f"Validated response type {type(validated_response)} is not supported. "
                    "Expected JSONResponse, ToolSelect, ToolSteps, TextResponse, or CodeResponse."
                )

        except Exception as e:
            logger.error(f"❌ [LLMManager] Generation failed: {e}", exc_info=True)
            raise RuntimeError(f"LLMManager generation failed: {e}") from e
