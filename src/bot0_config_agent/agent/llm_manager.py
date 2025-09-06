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
from typing import Any, Optional, Literal, Sequence, Union, Type, Tuple
import re
import json
import gc
from pydantic import BaseModel, ValidationError
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM
from llama_cpp import Llama
from gptqmodel import GPTQModel  # Ensure GPTQModel is installed or available
from awq import AutoAWQForCausalLM

# From project modules
from bot0_config_agent.loaders.model_configs_models import (
    TransformersLoaderConfig,
    LlamaCppLoaderConfig,
    AWQLoaderConfig,
    GPTQLoaderConfig,
    LoaderConfigEntry,
)
from bot0_config_agent.loaders.load_model_configs import load_model_configs
from bot0_config_agent.agent_models.agent_models import (
    JSONResponse,
    CodeResponse,
    TextResponse,
    ToolCall,
    ToolChain,
)
from bot0_config_agent.utils.llm.llm_response_validators import (
    validate_intent_response,
    validate_response_type,
    validate_tool_selection_or_steps,
)
from bot0_config_agent.utils.system.find_root_dir import find_project_root
from bot0_config_agent.utils.system.gpu_monitor import (
    log_gpu_usage,
    log_peak_vram_usage,
    log_embedding_footprint,
)
from bot0_config_agent.utils.llm.llm_prompt_payload_logger import (
    log_prompt_dict,
    log_llm_payload,
)


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
    Shared singleton getter (global cache to avoid reloading model into VRAM).

    #! Without this step, VRAM will overload!
    """
    if model_name not in _LLM_MANAGER_CACHE:
        _LLM_MANAGER_CACHE[model_name] = LLMManager(model_name)
    return _LLM_MANAGER_CACHE[model_name]


# def _map_dtype(s: str | None):
#     if not s:
#         return None
#     s = s.lower()
#     return {
#         "float16": torch.float16,
#         "bfloat16": torch.bfloat16,
#         "float32": torch.float32,
#     }.get(s)


# prompt_logger.py (add these)
def log_messages(logger, messages, level=logging.INFO, mode="yaml"):
    log_llm_payload(
        logger,
        label="LLM Messages",
        payload={"messages": messages},
        mode=mode,
        level=level,
    )


def log_output(logger, label, text, level=logging.DEBUG, mode="yaml"):
    log_llm_payload(
        logger,
        label=label,
        payload={"raw_output": text},
        mode=mode,
        level=level,
    )


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

        entry = load_model_configs(model_name)
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
        xml_tag: Optional[Union[str, Sequence[str]]] = None,
    ) -> Union[JSONResponse, TextResponse, CodeResponse, ToolCall, ToolChain]:
        """
        Generate a response using the loaded model. Expects output can be a JSON
        array of tool calls.

        Delegates to engine-specific _generate_with_*() method.
        The engine is responsible for handling output formatting (e.g., JSON parsing).

        Args:
            user_prompt: The user or system prompt to send to the model.
            system_prompt: The system prompt to send to the model (role, behavior, etc.)
            expected_res_type: The desired output type ("json", "text", "code", etc.).
            response_model: (Optional) Pydantic model to validate response
                (e.g., ToolChain).
            xml_tag (Optional[Union[str, Sequence[str]]]): Optional xml_tag for text or
                code response, such as "result" -> parses content inside <result>...</result>.
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

        logger.info("[LLMManager] Messages:")

        # Debugging
        log_prompt_dict(
            logger,
            label="[LLMManager] Messages",
            system_prompt=messages[0].get("content", ""),
            user_prompt=messages[1].get("content", ""),
            mode="yaml",  # or "json" if you prefer
            level=logging.INFO,
        )
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

            # Validation 1: response_type (code, text, json)
            # (& xml_tag if it's text or code)
            validated_response = validate_response_type(
                response, expected_res_type, xml_tag=xml_tag
            )

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

    def _map_dtype(self, s):
        import torch

        if not s:
            return None
        s = str(s).lower()
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(s)

    def _coerce_max_memory(self, mm: dict | None) -> dict | None:
        """
        Normalize YAML-provided max_memory for HF/Accelerate:
        keys: "0"->0, keep "cpu", also accept "cuda:0" -> 0.
        values: pass through (assumed '3.5GiB' etc.) — tighten if you want.
        """
        if not mm:
            return mm
        fixed = {}
        for k, v in mm.items():
            key = str(k).strip().lower()
            if key.isdigit():
                key = int(key)
            elif key.startswith("cuda:") and key[5:].isdigit():
                key = int(key[5:])
            elif key != "cpu":
                raise ValueError(
                    f"max_memory key must be GPU index or 'cpu', got {k!r}"
                )
            fixed[key] = str(v).strip()
        return fixed

    def _extract_model_id(self, config: BaseModel) -> tuple[str, dict]:
        """
        Accept a Pydantic loader config model (TransformersLoaderConfig, GPTQLoaderConfig,
        AWQLoaderConfig, or LlamaCppLoaderConfig) and return:
            (model_id_or_path, remaining_load_kwargs_dict)

        remaining_load_kwargs_dict are the HF-style loader kwargs (minus the id) you
        can pass through to your *_merge_* helpers.
        """

        # 1) Flat configs (e.g., llama_cpp) expose model_id_or_path directly
        mid = getattr(config, "model_id_or_path", None)
        if mid:
            return str(mid), {}  # llama.cpp path has no separate load_config kwargs

        # 2) HF-style loaders have nested load_config (Pydantic model or dict)
        load_cfg = getattr(config, "load_config", None)
        if load_cfg is None:
            raise ValueError(
                "Missing 'load_config' on HF-style config and no flat 'model_id_or_path' found."
            )

        # Normalize to dict
        if hasattr(load_cfg, "model_dump"):
            load_dict = load_cfg.model_dump()
        elif isinstance(load_cfg, dict):
            load_dict = dict(load_cfg)
        else:
            # Very defensive: generic attribute scrape (shouldn't be needed for Pydantic)
            load_dict = {
                k: getattr(load_cfg, k)
                for k in dir(load_cfg)
                if not k.startswith("_") and not callable(getattr(load_cfg, k))
            }

        # 3) Extract id/path (support both common key names)
        for key in ("model_id_or_path", "model_path"):
            val = load_dict.get(key)
            if val:
                mid = str(val)
                rest = {
                    k: v for k, v in load_dict.items() if k != key and v is not None
                }
                return mid, rest

        raise ValueError(
            "Missing required 'model_id_or_path' (or 'model_path') in config.load_config."
        )

    def _merge_awq_kwargs(self, load_cfg: dict, quant_cfg: dict | None) -> dict:
        """
        Merge load_config + quant_config for AutoAWQForCausalLM.from_quantized.
        Adds a sensible default: fuse_layers=True (override in YAML if needed).
        """
        allowed = {
            "device_map",
            "max_memory",
            "offload_folder",
            "torch_dtype",
            "trust_remote_code",
            "fuse_layers",
        }
        merged: dict = {}

        # from load_config
        for k, v in (load_cfg or {}).items():
            if k in ("model_id_or_path", "device"):  # handled elsewhere
                continue
            if k == "max_memory":
                v = self._coerce_max_memory(v)
            if k == "torch_dtype":
                v = self._map_dtype(v)
            if k in allowed and v is not None:
                merged[k] = v

        # from quant_config (e.g., fuse_layers)
        for k, v in (quant_cfg or {}).items():
            if k in allowed and v is not None:
                merged[k] = v

        merged.setdefault("fuse_layers", True)  # default helpful for AWQ
        return {k: v for k, v in merged.items() if v is not None}

    def _merge_gptq_kwargs(self, load_cfg: dict, quant_cfg: dict | None) -> dict:
        """
        Merge load_config + quant_config for GPTQModel.from_quantized/from_pretrained.
        (No fuse_layers here.)
        """
        allowed = {
            "device_map",
            "max_memory",
            "offload_folder",
            "torch_dtype",
            "trust_remote_code",
        }
        merged: dict = {}

        for k, v in (load_cfg or {}).items():
            if k in ("model_id_or_path", "device"):
                continue
            if k == "max_memory":
                v = self._coerce_max_memory(v)
            if k == "torch_dtype":
                v = self._map_dtype(v)
            if k in allowed and v is not None:
                merged[k] = v

        for k, v in (quant_cfg or {}).items():
            if k in allowed and v is not None:
                merged[k] = v

        return {k: v for k, v in merged.items() if v is not None}

    def _merge_transformers_kwargs(self, load_cfg: dict) -> dict:
        """Merge kwargs for plain HF Transformers models (no quant config)."""
        allowed = {
            "device_map",
            "torch_dtype",
            "trust_remote_code",
            "max_memory",
        }
        merged: dict = {}
        for k, v in (load_cfg or {}).items():
            if k in ("model_id_or_path",):
                continue
            if k == "max_memory":
                v = self._coerce_max_memory(v)
            if k in allowed:
                merged[k] = v
        return merged

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

        Extract path or id from pydantic model and returns separated dict.
        Combine load_config and quant_config (if needed)
        Load the model & load the tokenizer.

        Args:
            config (AWQLoaderConfig):
                Configuration object for the AWQ model, including model path, device,
                dtype, and other loader-specific options.
        """
        # 1) Get model id and normalized loader kwargs (dict)
        model_id, load_cfg = self._extract_model_id(
            config
        )  # <— now accepts Pydantic model

        # 2) Quant kwargs (dict)
        quant_cfg = getattr(config, "quant_config", None)
        if quant_cfg is not None and hasattr(quant_cfg, "model_dump"):
            quant_cfg = quant_cfg.model_dump()
        elif quant_cfg is None:
            quant_cfg = {}

        # 3) Merge loader + quant kwargs (no explicit device; AWQ/accelerate handles placement)
        kwargs = self._merge_awq_kwargs(load_cfg, quant_cfg)
        kwargs.setdefault(
            "device_map", "auto"
        )  # default shard/offload if not specified

        # 4) Load model
        self.model = AutoAWQForCausalLM.from_quantized(model_id, **kwargs)

        # 5) Tokenizer (respect trust_remote_code from either merged kwargs or load_cfg)
        trust_remote = bool(
            kwargs.get("trust_remote_code", load_cfg.get("trust_remote_code", False))
        )
        tok = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=trust_remote
        )
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        self.tokenizer = tok

    def _load_model_with_gptq(self, config: GPTQLoaderConfig) -> None:
        """
        Load a GPTQ-quantized model from a Pydantic loader config.

        What this does:
        - Extracts the positional `model_id_or_path` from the Pydantic config
        (supports both flat configs and nested `load_config`).
        - Normalizes `load_config` and `quant_config` (Pydantic -> dict).
        - Merges them into GPTQ loader kwargs via `_merge_gptq_kwargs(...)`.
        - Normalizes sharding/offload behavior:
            * If `device_map` or `max_memory` is present → remove any explicit `device`.
            * Otherwise, set single-device `device` (cuda if available, else cpu).
        - Loads the model with `GPTQModel.from_quantized(model_id, **kwargs)`.
        - Initializes a tokenizer and ensures `pad_token_id` is set.

        Args:
            config (GPTQLoaderConfig): Validated GPTQ loader configuration model.
        """
        # 1) Extract model_id and normalized loader kwargs (dict)
        model_id, load_cfg = self._extract_model_id(
            config
        )  # accepts the Pydantic model

        # 2) Quant kwargs (dict)
        quant_cfg = getattr(config, "quant_config", None)
        if quant_cfg is not None and hasattr(quant_cfg, "model_dump"):
            quant_cfg = quant_cfg.model_dump()
        elif quant_cfg is None:
            quant_cfg = {}

        # 3) Merge kwargs for GPTQ (also where you normalize max_memory keys, etc.)
        kwargs = self._merge_gptq_kwargs(load_cfg, quant_cfg)

        # Consider both sharding and offload as "multi-device" → don't pass `device`
        has_sharding = (
            "device_map" in kwargs and kwargs["device_map"] is not None
        ) or ("max_memory" in kwargs and kwargs["max_memory"])
        if has_sharding:
            kwargs.pop("device", None)
        else:
            kwargs.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")

        # 4) Load model
        self.model = GPTQModel.from_quantized(model_id, **kwargs)

        # 5) Tokenizer (respect trust_remote_code from either merged kwargs or load_cfg)
        trust_remote = bool(
            kwargs.get("trust_remote_code", load_cfg.get("trust_remote_code", False))
        )
        tok = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=trust_remote
        )
        assert isinstance(tok, PreTrainedTokenizerBase)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        self.tokenizer = tok

    def _load_model_with_llama_cpp(self, config: LlamaCppLoaderConfig) -> None:
        """
        Load a GGUF llama.cpp model.

        Unlike other loaders (Transformers, GPTQ, AWQ) which require us to
        normalize configs (pull out `model_id_or_path`, merge into `load_config`,
        call `_extract_model_id`, etc.), llama.cpp already expects a clean
        keyword-only constructor.

        Therefore, pass the Pydantic config fields directly to `Llama(...)`,
        with only one remapping: `model_id_or_path` → `model_path`. No additional
        normalization, merging, or "positional-to-keyword" conversion is needed.

        In practice:
            - YAML → Pydantic model → `model_dump()` → kwargs
            - Rename `model_id_or_path` to `model_path`
            - Pass kwargs directly into llama-cpp

        This preserves the YAML as-is, so your llama.cpp configs can be written
        naturally without worrying about adapter code.

        Args:
            config (LlamaCppLoaderConfig): GGUF config Pydantic model,
                containing fields like `model_id_or_path`, `n_ctx`,
                `n_gpu_layers`, `chat_format`, `verbose`, etc.
        """
        kwargs = config.load_config

        print(f"llama_cpp kwargs: {kwargs}")  # todo: debug; delete later

        if "model_id_or_path" not in kwargs:
            raise ValueError("Missing model_id_or_path in config for llama-cpp")

        kwargs["model_path"] = kwargs.pop(
            "model_id_or_path"
        )  # llama-cpp only takes model_path

        self.model = Llama(**kwargs)
        self.tokenizer = self.model

    def _load_model_with_transformers(self, config: TransformersLoaderConfig) -> None:
        """
        Load a standard HF Transformers causal LM from a Pydantic loader config.

        What this does:
        - Accepts a `TransformersLoaderConfig` (with nested `load_config`).
        - Extracts the positional `model_id_or_path` directly from the Pydantic model
        via `_extract_model_id(config)`.
        - Normalizes `load_config` into a plain dict and merges kwargs with
        `_merge_transformers_kwargs(...)`.
        - Defaults to `device_map="auto"` if not provided (enables sharding/offload).
        - Loads `AutoModelForCausalLM` and a matching `AutoTokenizer`, ensuring
        `pad_token_id` is set.

        Args:
            config (TransformersLoaderConfig): Validated transformers loader configuration.
        """
        # 1) Extract model id and normalized loader kwargs (dict)
        model_id, load_cfg = self._extract_model_id(config)

        # 2) Merge loader kwargs for transformers
        kwargs = self._merge_transformers_kwargs(load_cfg)
        kwargs.setdefault("device_map", "auto")

        # 3) Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        # 4) Tokenizer (respect trust_remote_code from either merged kwargs or load_cfg)
        trust_remote = bool(
            kwargs.get("trust_remote_code", load_cfg.get("trust_remote_code", False))
        )
        tok = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=trust_remote
        )
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        self.tokenizer = tok

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
        try:
            model_id, _ = self._extract_model_id(config)
            model_name = model_id
        except Exception:
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

        # ---- ✅ embedding-specific checkpoint (generic across all loaders) ----
        # * This is for log embedding VRAM usage
        if getattr(self, "model", None) is not None and self.loader != "llama_cpp":
            try:
                # totals across all embedding modules
                log_embedding_footprint(self.model, label=f"{model_name}")

                # optionally, confirm the input embedding exists and log again
                emb = getattr(self.model, "get_input_embeddings", lambda: None)()
                if emb is not None:
                    log_embedding_footprint(
                        self.model, label=f"{model_name}:input_emb_only"
                    )
            except Exception as e:
                logger.debug(f"[LLMManager] Embedding logging skipped: {e}")

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
        log_llm_payload(
            logger,
            label="[AWQ] 🧪 Raw decoded output",
            payload={"output": decoded},
            mode="yaml",
            level=logging.DEBUG,
        )

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
        log_llm_payload(
            logger,
            label="[GPTQ] 🧪 Raw decoded output",
            payload={"output": decoded},
            mode="yaml",
            level=logging.DEBUG,
        )

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

        # logger.debug(f"[llama_cpp] 🧪 Raw output:\n{repr(content)}")
        # Debug in nice foramt
        log_llm_payload(
            logger,
            label="[llama_cpp] 🧪 Raw output",
            payload={"output": content},
            mode="yaml",  # or "text" if you want plain
            level=logging.DEBUG,
        )

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
        log_llm_payload(
            logger,
            label="[Transformers] 🧪 Raw decoded output",
            payload={"output": decoded},
            mode="yaml",
            level=logging.DEBUG,
        )

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
