# agent/llm_manager.py
# CORRECTED: Replaced all instances of the incorrect "HUGGINGFACE_TOKEN"
# with the correct "HUGGING_FACE_HUB_TOKEN" environment variable.

import logging
import os
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Union, Type, Tuple, List
from pydantic import BaseModel, ValidationError
import re
import json
import torch
import gc
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForCausalLM
from llama_cpp import Llama
from gptqmodel import GPTQModel
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
import asyncio

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

# ===== NEW: Sentinel and robust JSON extraction helpers =====
_SENTINEL = "FINAL_JSON"

def _strip_code_fences(s: str) -> str:
    """
    Remove common markdown code fences around content to reduce extraction failures.
    """
    s = re.sub(r"```(?:json|JSON)?\s*", "", s)
    s = s.replace("```", "")
    return s

def _last_sentinel_index(s: str) -> int:
    """
    Find the last occurrence of a LINE that is exactly FINAL_JSON (ignoring surrounding spaces).
    Returns the index *after* the sentinel line, or -1 if not found.
    """
    matches = list(re.finditer(r"(?mi)^\s*FINAL_JSON\s*$", s))
    if not matches:
        return -1
    m = matches[-1]
    return m.end()

def _scan_balanced_json_array(s: str, start_pos: int) -> Optional[str]:
    """
    From start_pos, find the first '[' and return the substring of the *balanced*
    JSON array including nested objects/arrays. Ignores brackets inside quoted strings.
    Returns None if not found or unbalanced.
    """
    n = len(s)
    i = start_pos
    while i < n and s[i] != "[":
        i += 1
    if i >= n:
        return None
    depth = 0
    in_string = False
    escape = False
    start_idx = i
    while i < n:
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return s[start_idx : i + 1]
        i += 1
    return None

def _find_all_top_level_arrays(s: str) -> List[str]:
    """
    Scan the whole string and return all well-formed top-level JSON arrays found.
    This ignores arrays inside strings and ensures brackets are balanced.
    """
    arrays = []
    n = len(s)
    i = 0
    while i < n:
        if s[i] == "[":
            arr = _scan_balanced_json_array(s, i)
            if arr:
                arrays.append(arr)
                i += len(arr)
                continue
        i += 1
    return arrays

def _extract_json_array_text(raw_text: str) -> str:
    """
    Preferred extraction:
      1) Look for last 'FINAL_JSON' line, then parse the first complete array after it.
      2) If no sentinel, strip code fences and gather all arrays; choose the last valid one.
      3) If nothing parseable, return "[]".
    """
    text = raw_text.strip()
    sentinel_pos = _last_sentinel_index(text)
    if sentinel_pos != -1:
        after = text[sentinel_pos:].lstrip()
        arr = _scan_balanced_json_array(after, 0)
        if arr:
            logger.debug("[LLMManager] ✅ Extracted JSON via sentinel FINAL_JSON.")
            return arr
        else:
            logger.warning(
                "[LLMManager] ⚠️ FINAL_JSON found but array was not well-formed right after it."
            )
    stripped = _strip_code_fences(text)
    candidates = _find_all_top_level_arrays(stripped)
    for cand in reversed(candidates):
        try:
            json.loads(cand)
            logger.debug("[LLMManager] ✅ Extracted JSON via fallback array scan.")
            return cand
        except Exception:
            continue
    logger.warning("[LLMManager] ⚠️ No valid JSON array found. Returning empty array.")
    return "[]"

# ===== end helpers =====

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
        self.model_name: str = model_name
        entry = load_model_config(model_name)
        self.loader = entry.loader
        config: (
            AWQLoaderConfig
            | GPTQLoaderConfig
            | LlamaCppLoaderConfig
            | TransformersLoaderConfig
        ) = entry.config
        self.generation_config: dict = (
            entry.generation_config or {}
        )
        logger.info(f"[LLMManager] 📦 Initializing model: {model_name} ({self.loader})")
        self._load_model(config)

    def cleanup_vram_cache(self):
        """
        Safely deletes the previous model, runs garbage collection,
        and empties the CUDA cache. Logs VRAM usage before and after.
        Returns None (the new value to assign to your model variable).
        """
        log_gpu_usage("[LLMManager] before clearing up vram cache")
        if hasattr(self, "model") and self.model is not None:
            logger.info("[LLMManager] Releasing old model from memory...")
            del self.model
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
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
        """
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        if expected_res_type not in ["json", "text", "code"]:
            raise ValueError(
                f"Invalid expected_res_type '{expected_res_type}'. "
                "Must be one of: 'json', 'text', or 'code'."
            )
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        gen_cfg = self.generation_config.copy()
        if not gen_cfg:
            raise ValueError("No generation config found for this model!")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(f"[LLMManager] Messages:\n{json.dumps(messages, indent=2)}\n")
        logger.debug(f"Generating with {self.loader}")
        torch.cuda.reset_peak_memory_stats()
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
                raise ValueError(f"Unsupported loader: {self.loader}")
            if not response:
                raise ValueError(f"Generated output is empty [].")
            after_peak_vram = log_peak_vram_usage(
                f"peak VRAM during generation with {self.model_name}"
            )
            vram_jump = after_peak_vram - before_vram
            logger.info(
                f"[LLMManager] VRAM jump for {self.model_name}: {vram_jump:.1f} MB "
                f"(before: {before_vram:.1f} MB, peak: {after_peak_vram})"
            )
            logger.info("[LLMManager] 🧪 Generated text (raw):\n%s", response)
            validated_response = validate_response_type(response, expected_res_type)
            if isinstance(validated_response, JSONResponse):
                validated_response_model = validated_response
                if response_model:
                    response_models = (response_model,) if not isinstance(response_model, tuple) else response_model
                    if any(m in (ToolCall, ToolChain) for m in response_models):
                        response_data = validated_response_model.data
                        try:
                            validated_response_model = validate_tool_selection_or_steps(response_data)
                        except ValidationError as ve:
                            logger.error("Tool selection validation failed: %s", ve)
                            raise ValueError(f"Tool selection validation failed: {ve}") from ve
                logger.info(f"validated response content after validate_json_type: \n{validated_response_model}")
                return validated_response_model
            elif isinstance(validated_response, (TextResponse, CodeResponse)):
                return validated_response
            else:
                logger.error(f"Validated response has unsupported type: {type(validated_response)}; Value: {repr(validated_response)}")
                raise TypeError(
                    f"Validated response type {type(validated_response)} is not supported. "
                    "Expected JSONResponse, ToolCall, ToolChain, TextResponse, or CodeResponse."
                )
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Validation or parsing error: {e}")
            raise ValueError(f"Invalid format received from loader '{self.loader}': {e}") from e
        except Exception as e:
            logger.error(f"{self.loader} generate() failed: {e}")
            raise RuntimeError(f"Model generation failed with loader '{self.loader}': {e}") from e

    async def generate_async(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        expected_res_type: Literal["json", "text", "code"] = "text",
        response_model: Optional[Type[BaseModel] | Tuple[Type[BaseModel], ...]] = None,
    ) -> Union[JSONResponse, TextResponse, CodeResponse, ToolCall, ToolChain]:
        """
        Asynchronously generate a response using the loaded model.
        Delegates to engine-specific sync generate methods via an executor.
        """
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        if expected_res_type not in ["json", "text", "code"]:
            raise ValueError(
                f"Invalid expected_res_type '{expected_res_type}'. "
                "Must be one of: 'json', 'text', or 'code'."
            )
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        gen_cfg = self.generation_config.copy()
        if not gen_cfg:
            raise ValueError("No generation config found for this model!")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(f"[LLMManager] Messages:\n{json.dumps(messages, indent=2)}\n")
        logger.debug(f"Generating async with {self.loader}")
        torch.cuda.reset_peak_memory_stats()
        before_vram = log_gpu_usage(f"before inference with {self.model_name}")
        try:
            loop = asyncio.get_event_loop()
            if self.loader == "awq":
                response = await loop.run_in_executor(
                    None,
                    lambda: self._generate_with_awq(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        expected_res_type=expected_res_type,
                        **gen_cfg,
                    )
                )
            elif self.loader == "gptq":
                response = await loop.run_in_executor(
                    None,
                    lambda: self._generate_with_gptq(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        expected_res_type=expected_res_type,
                        **gen_cfg,
                    )
                )
            elif self.loader == "llama_cpp":
                response = await loop.run_in_executor(
                    None,
                    lambda: self._generate_with_llama_cpp(
                        messages=messages,
                        expected_res_type=expected_res_type,
                        **gen_cfg,
                    )
                )
            elif self.loader == "transformers":
                response = await loop.run_in_executor(
                    None,
                    lambda: self._generate_with_transformers(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        expected_res_type=expected_res_type,
                        **gen_cfg,
                    )
                )
            else:
                raise ValueError(f"Unsupported loader: {self.loader}")
            if not response:
                raise ValueError(f"Generated output is empty [].")
            after_peak_vram = log_peak_vram_usage(f"peak VRAM during generation with {self.model_name}")
            vram_jump = after_peak_vram - before_vram
            logger.info(
                f"[LLMManager] VRAM jump for {self.model_name}: {vram_jump:.1f} MB "
                f"(before: {before_vram:.1f} MB, peak: {after_peak_vram})"
            )
            logger.info("[LLMManager] 🧪 Generated text (raw):\n%s", response)
            validated_response = validate_response_type(response, expected_res_type)
            if isinstance(validated_response, JSONResponse):
                validated_response_model = validated_response
                if response_model:
                    response_models = (response_model,) if not isinstance(response_model, tuple) else response_model
                    if any(m in (ToolCall, ToolChain) for m in response_models):
                        response_data = validated_response_model.data
                        try:
                            validated_response_model = validate_tool_selection_or_steps(response_data)
                        except ValidationError as ve:
                            logger.error("Tool selection validation failed: %s", ve)
                            raise ValueError(f"Tool selection validation failed: {ve}") from ve
                logger.info(f"validated response content after validate_json_type: \n{validated_response_model}")
                return validated_response_model
            elif isinstance(validated_response, (TextResponse, CodeResponse)):
                return validated_response
            else:
                logger.error(f"Validated response has unsupported type: {type(validated_response)}; Value: {repr(validated_response)}")
                raise TypeError(
                    f"Validated response type {type(validated_response)} is not supported. "
                    "Expected JSONResponse, ToolCall, ToolChain, TextResponse, or CodeResponse."
                )
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Validation or parsing error: {e}")
            raise ValueError(f"Invalid format received from loader '{self.loader}': {e}") from e
        except Exception as e:
            logger.error(f"{self.loader} generate_async() failed: {e}")
            raise RuntimeError(f"Model generation failed with loader '{self.loader}': {e}") from e

    def _load_model_with_awq(self, config: AWQLoaderConfig) -> None:
        """
        Load an AWQ quantized model using AutoAWQForCausalLM.from_quantized.
        """
        if getattr(config, "device", None) in (None, "auto"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        if self.device == "cpu":
            logger.warning("[LLMManager] ⚠️ AWQ model will run on CPU — this may be very slow or unsupported.")
        loader_kwargs = config.model_dump(exclude={"model_id_or_path", "device"})
        self.model = AutoAWQForCausalLM.from_quantized(
            config.model_id_or_path,
            device=self.device,
            fuse_layers=False,
            **loader_kwargs,
        )
        try:
            param_device = next(self.model.parameters()).device
            logger.info(f"[LLMManager] AWQ model param device: {param_device}")
            if str(param_device) != self.device and self.device != "auto":
                logger.warning(f"[LLMManager] MISMATCH: Model param is on {param_device}, expected {self.device}")
                logger.info("[LLMManager] Attempting to move model to correct device...")
                self.model = self.model.to(self.device)
                param_device = next(self.model.parameters()).device
                logger.info(f"[LLMManager] Model param device after .to(): {param_device}")
        except Exception as e:
            logger.warning(f"[LLMManager] Could not check or move model param device: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id_or_path, use_fast=True, token=os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def _load_model_with_gptq(self, config: GPTQLoaderConfig) -> None:
        """
        Load a GPTQ quantized model.
        """
        loader_kwargs = config.model_dump(exclude={"model_id_or_path"})
        loader_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        loader_kwargs.pop("disable_exllama", None)
        loader_kwargs.pop("group_size", None)  # <-- ADD THIS NEW LINE
        self.model = GPTQModel.from_quantized(config.model_id_or_path, **loader_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id_or_path, use_fast=True, token=os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def _load_model_with_llama_cpp(self, config: LlamaCppLoaderConfig) -> None:
        """
        Load a GGUF llama.cpp model.
        """
        loader_kwargs = config.model_dump(exclude={"model_id_or_path"})
        loader_kwargs["model_path"] = config.model_id_or_path
        self.model = Llama(**loader_kwargs)
        self.tokenizer = self.model

    def _load_model_with_transformers(self, config: TransformersLoaderConfig) -> None:
        """
        Load a standard Transformers model from Hugging Face hub or local path.
        """
        loader_kwargs = config.model_dump(exclude={"model_id_or_path"})
        device = loader_kwargs.pop("device", None)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id_or_path,
            token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
            **loader_kwargs,
        )
        if device:
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id_or_path, use_fast=True, token=os.getenv("HUGGING_FACE_HUB_TOKEN")
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
        """
        self.cleanup_vram_cache()
        self.device = getattr(
            config, "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_name = getattr(config, "model_id_or_path", "unknown")
        logger.info(
            f"[LLMManager] ✅ Using model: {model_name} ({self.loader}) on {self.device}"
        )
        log_gpu_usage(f"[LLMManager] before loading model {model_name}")
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
        log_gpu_usage(f"[LLMManager] after loading model {model_name}")

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
        """
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerBase
        ), "Expected a Hugging Face tokenizer for AWQ models."
        full_prompt = self._format_prompt(user_prompt, system_prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        logger.debug(f"[AWQ] 🔁 Prompt:\n{full_prompt}")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=generation_kwargs.get("temperature", 0.3) > 0.0,
                **generation_kwargs,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"[AWQ] 🧪 Decoded output:\n{repr(decoded)}")
        if expected_res_type == "json":
            return _extract_json_array_text(decoded)
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
            return _extract_json_array_text(decoded)
        return decoded

    def _generate_with_llama_cpp(
        self,
        messages: list,
        expected_res_type: Literal["json", "text", "code"] = "text",
        **generation_kwargs,
    ) -> str:
        """
        Generates a response using a GGUF model loaded via llama.cpp.
        """
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        logger.debug(
            f"[llama_cpp] 🔁 Input messages:\n{json.dumps(messages, indent=2)}"
        )
        kwargs = {
            "messages": messages,
            "max_tokens": generation_kwargs.get("max_tokens", 256),
            "temperature": generation_kwargs.get("temperature", 0.2),
            "top_p": generation_kwargs.get("top_p", 0.95),
            "top_k": generation_kwargs.get("top_k", 40),
            "stop": generation_kwargs.get("stop", ["</s>"]),
        }
        if "top_k" in kwargs:
            kwargs["top_k"] = int(kwargs["top_k"])
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        output = self.model.create_chat_completion(**kwargs)
        content = output["choices"][0]["message"]["content"].strip()
        logger.debug(f"[llama_cpp] 🧪 Raw output:\n{repr(content)}")
        if expected_res_type == "json":
            return _extract_json_array_text(content)
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
            return _extract_json_array_text(decoded)
        return decoded

