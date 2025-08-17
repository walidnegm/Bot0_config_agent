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
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Union, Type, Tuple
from pydantic import BaseModel, ValidationError
import re
import json
import torch
import gc
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
from agent_models.agent_models import (
    JSONResponse,
    CodeResponse,
    TextResponse,
    ToolCall,
    ToolChain,
)
from agent_models.llm_response_validators import (
    validate_response_type,
    validate_tool_selection_or_steps,
)
from utils.find_root_dir import find_project_root
from utils.gpu_monitor import log_gpu_usage, log_peak_vram_usage


logger = logging.getLogger(__name__)

try:
    root_dir = find_project_root()
except Exception as e:
    raise FileNotFoundError(
        "❌ Could not determine project root. Make sure one of the expected markers exists \
(e.g., .git, requirements.txt, pyproject.toml, README.md)."
    ) from e

ModelLoaderType = Literal["awq", "gptq", "llama_cpp", "transformers"]

_LLM_MANAGER_CACHE = {}


#! Single instance of model
def get_llm_manager(model_name):
    """
    Shared singleton getter (global cache to avoid reloading model into VRAM)
    """
    if model_name not in _LLM_MANAGER_CACHE:
        _LLM_MANAGER_CACHE[model_name] = LLMManager(model_name)
    return _LLM_MANAGER_CACHE[model_name]


class LLMManager:
    """
    Loads and manages local LLMs (GPTQ, GGUF, or AWQ) for inference, with prompt formatting
    and generation support.
    """

    def __init__(self, model_name: str):
        """
        Initialize the LLMManager.

        Args:
            model_name (str): model name (for looking up model in config files)
        """
        self.loader: Optional[ModelLoaderType] = None
        self.tokenizer: Optional[Union[PreTrainedTokenizerBase, Llama]] = None
        self.model: Optional[Any] = None
        self.model_name: str = model_name  # For logging

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

    def cleanup_vram_cache(self):
        """
        Safely deletes the previous model, runs garbage collection,
        and empties the CUDA cache. Logs VRAM usage before and after.
        Returns None (the new value to assign to your model variable).
        """
        # Log VRAM before cleanup
        log_gpu_usage("[LLMMager] before clearing up vram cache")

        if hasattr(self, "model") and self.model is not None:
            logger.info("[LLMManager] Releasing old model from memory...")
            del self.model
            self.model = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Log VRAM after cleanup
        log_gpu_usage("[LLMManager] after clearing up VRAM cache")

        return None

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        expected_res_type: Literal["json", "text", "code"] = "text",
        response_model: Optional[Type[BaseModel] | Tuple[Type[BaseModel], ...]] = None,
    ) -> Union[JSONResponse, TextResponse, CodeResponse, ToolCall, ToolChain]:
        """
        Generate a response using the loaded model. Expects output can be a JSON
        array of tool calls.

        Delegates to engine-specific _generate_with_*() method.
        The engine is responsible for handling output formatting (e.g., JSON parsing).
        """
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."

        # Validate response type before returning
        if expected_res_type not in ["json", "text", "code"]:
            raise ValueError(
                f"Invalid expected_res_type '{expected_res_type}'. "
                "Must be one of: 'json', 'text', or 'code'."
            )
        # System prompt safeguard
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        # Load generation config from class
        gen_cfg = self.generation_config.copy()
        if not gen_cfg:
            raise ValueError("No generation config found for this model!")

        # Special setting for llama-cpp
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"[LLMManager] Messages:\n{json.dumps(messages, indent=2)}\n")
        logger.debug(f"Generating with {self.loader}")

        # ☑️ log VRAM before inference
        torch.cuda.reset_peak_memory_stats()  # clear any earlier VRAM peaks
        before_vram = log_gpu_usage(f"before inference with {self.model_name}")

        try:
            if self.loader == "awq":
                response = self._generate_with_awq(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    expected_res_type=expected_res_type,
                    **gen_cfg,
                )

            elif self.loader == "gptq":
                response = self._generate_with_gptq(
                    user_prompt=user_prompt,
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
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    expected_res_type=expected_res_type,
                    **gen_cfg,
                )

            else:
                raise ValueError(f"Unsupported loader type: {self.loader}")

            if not response:
                raise ValueError(f"Generated output is empty [].")

            # ☑️ log VRAM usage during inference
            after_peak_vram = log_peak_vram_usage(
                f"peak VRAM during generation with {self.model_name}"
            )  # ☑️ track peak vram usage
            vram_jump = after_peak_vram - before_vram
            logger.info(
                f"[LLMManager] VRAM jump for {self.model_name}: {vram_jump:.1f} MB "
                f"(before: {before_vram:.1f} MB, peak: {after_peak_vram})"
            )

            # * Critical logging to examine main output
            logger.info("[LLMManager] 🧪 Generated text (raw):\n%s", response)
            # logger.info(
            #     "[LLMManager] 🧪 Generated text (pretty):\n%s",
            #     (
            #         json.dumps(json.loads(response), indent=2, ensure_ascii=False)
            #         if isinstance(response, str)
            #         and response.strip().startswith(("[", "{"))
            #         else response
            #     ),
            # )

            # Validation 1: response_type (code, text, json)
            validated_response = validate_response_type(response, expected_res_type)

            # Handle JSON responses that may be ToolCall or ToolChain
            if isinstance(validated_response, JSONResponse):
                validated_response_model = (
                    validated_response  # <-- set as a new var to be safe
                )
                if response_model:
                    # * Normalize to tuple for flexible membership testing b/c response_model
                    # * can be a single model or list of models
                    if not isinstance(response_model, tuple):
                        response_models = (response_model,)
                    else:
                        response_models = response_model

                    # Custom logic: If any response_model is ToolCall or ToolChain
                    if any(m in (ToolCall, ToolChain) for m in response_models):
                        response_data = validated_response_model.data
                        try:
                            validated_response_model = validate_tool_selection_or_steps(
                                response_data
                            )
                        except ValidationError as ve:
                            logger.error("Tool selection validation failed: %s", ve)
                            raise ValueError(
                                f"Tool selection validation failed: {ve}"
                            ) from ve
                logger.info(
                    f"validated response content after validate_json_type: \n{validated_response_model}"
                )
                return validated_response_model

            elif isinstance(validated_response, (TextResponse, CodeResponse)):
                return validated_response

            else:
                logger.error(
                    f"Validated response has unsupported type: {type(validated_response)}; "
                    f"Value: {repr(validated_response)}"
                )
                raise TypeError(
                    f"Validated response type {type(validated_response)} is not supported. "
                    "Expected JSONResponse, ToolCall, ToolChain, TextResponse, or CodeResponse."
                )

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Validation or parsing error: {e}")
            raise ValueError(
                f"Invalid format received from loader '{self.loader}': {e}"
            ) from e
        except Exception as e:
            logger.error(f"{self.loader} generate() failed: {e}")
            raise RuntimeError(
                f"Model generation failed with loader '{self.loader}': {e}"
            ) from e

    def _load_model_with_awq(self, config: AWQLoaderConfig) -> None:
        """
        Load an AWQ quantized model using AutoAWQForCausalLM.from_quantized.

        This method ensures that:
        - The `device` is properly resolved (defaults to 'cuda' if available for
            AWQ models).
        - The model is initialized with the positional `model_id_or_path` argument
            passed explicitly.
        This is necessary because `from_quantized()` requires `model_id_or_path`
        as a positional parameter, and we avoid including it in `**loader_kwargs`
        to prevent accidental duplication or argument conflicts.

        Args:
            config (AWQLoaderConfig):
                Configuration object for the AWQ model, including model path, device,
                dtype, and other loader-specific options.
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

        # --- ENFORCE ALL MODEL PARAMS ARE ON THE SAME DEVICE ---
        try:
            param_device = next(self.model.parameters()).device
            logger.info(f"[LLMManager] AWQ model param device: {param_device}")
            if str(param_device) != self.device and self.device != "auto":
                logger.warning(
                    f"[LLMManager] MISMATCH: Model param is on {param_device}, expected {self.device}"
                )
                logger.info(
                    "[LLMManager] Attempting to move model to correct device..."
                )
                self.model = self.model.to(
                    self.device
                )  # Will work if model supports .to()
                param_device = next(self.model.parameters()).device
                logger.info(
                    f"[LLMManager] Model param device after .to(): {param_device}"
                )
        except Exception as e:
            logger.warning(
                f"[LLMManager] Could not check or move model param device: {e}"
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
        device = loader_kwargs.pop("device", None)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id_or_path,
            **loader_kwargs,
        )

        if device:
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)

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
        self.cleanup_vram_cache()  # Clear up VRAM first!

        # Set device from config if available (BaseHFModelConfig), else infer
        self.device = getattr(
            config, "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Get the model name for logging
        model_name = getattr(config, "model_id_or_path", "unknown")

        logger.info(
            f"[LLMManager] ✅ Using model: {model_name} ({self.loader}) on {self.device}"
        )

        log_gpu_usage(f"[LLMManager] before loading model {model_name}")  # ☑️ track vram

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

        log_gpu_usage(f"[LLMManager] after loading model {model_name}")  # ☑️ track vram

    def _format_prompt(self, user_prompt: str, system_prompt: str = "") -> str:
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    def _generate_with_awq(
        self,
        user_prompt: str,
        system_prompt: str,
        expected_res_type: Literal["json", "text", "code"] = "text",
        **generation_kwargs,
    ) -> str:
        """
        Generates a response using autoawq library.

        Args:
            user_prompt (str): User prompt.
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

        full_prompt = self._format_prompt(user_prompt, system_prompt)
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
        user_prompt: str,
        system_prompt: str,
        expected_res_type: Literal["json", "text", "code"] = "text",
        **generation_kwargs,
    ) -> str:
        """
        Generate a response from a GPTQ-quantized Hugging Face model.

        Args:
            user_prompt (str): The user input prompt.
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

        full_prompt = self._format_prompt(user_prompt, system_prompt)
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
        expected_res_type: Literal["json", "text", "code"] = "text",
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
        user_prompt: str,
        system_prompt: str,
        expected_res_type: Literal["json", "text", "code"] = "text",
        **generation_kwargs,
    ) -> str:
        """
        Generate a response using a standard Transformers model.

        Args:
            user_prompt (str): The user input prompt.
            system_prompt (str): System-level behavior instruction.
            expected_res_type (str | None): Optionally extract JSON from output.
            **generation_kwargs: All generation params (max_new_tokens, temperature, etc.)

        Returns:
            str: Raw or extracted JSON response.
        """
        assert self.model is not None and self.tokenizer is not None
        assert isinstance(self.tokenizer, PreTrainedTokenizerBase)

        full_prompt = self._format_prompt(user_prompt, system_prompt)
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
