"""
agent/llm_manager.py
Unified manager for calling LLMs (local or API).
- Local: Uses transformers or llama_cpp_python, with config from configs/model_configs.yaml
- API: Uses OpenAIAdapter for API-backed models (OpenAI, Gemini, etc.)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from configs.paths import MODEL_CONFIGS_YAML_FILE
from configs.api_models import get_llm_provider, validate_api_model_name
from agent.llm_openai import OpenAIAdapter
import torch
from loaders.model_configs_models import TransformersLoaderConfig

logger = logging.getLogger(__name__)

class LLMInitConfig(BaseModel):
    name: str
    loader: str
    hf_model: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    trust_remote_code: bool = True
    api_model: Optional[str] = None

class LLMManager:
    def __init__(self, local_model: Optional[str] = None, api_model: Optional[str] = None):
        """
        Initialize LLMManager for either a local or API model.

        Args:
            local_model (Optional[str]): Name of local LLM model (from configs/model_configs.yaml).
            api_model (Optional[str]): Name of API model (e.g., gpt-4o, gemini-1.5-pro).

        Raises:
            ValueError: If both or neither model is specified, or if the local model config is invalid.
            ImportError: If required dependencies are missing.
        """
        if (local_model and api_model) or (not local_model and not api_model):
            raise ValueError("Exactly one of local_model or api_model must be specified.")

        self.local_model = local_model
        self.api_model = api_model
        self.model_name = local_model or api_model or ""
        self._client = None

        if local_model:
            try:
                from loaders.load_model_config import load_model_config
            except ImportError as e:
                logger.error("Failed to import load_model_config; ensure loaders are installed.")
                raise ImportError("Failed to import load_model_config; ensure loaders are installed.") from e

            try:
                proj_cfg = load_model_config(local_model, config_file=MODEL_CONFIGS_YAML_FILE)
                logger.info(f"Loaded config for '{local_model}': loader={proj_cfg.loader}, config_type={type(proj_cfg.config).__name__}, config={proj_cfg.config}")
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Failed to load model configuration for '{local_model}' from {MODEL_CONFIGS_YAML_FILE}: {e}")
                raise

            # Only access trust_remote_code for configs that support it
            trust_remote_code = True
            if isinstance(proj_cfg.config, TransformersLoaderConfig):
                trust_remote_code = proj_cfg.config.trust_remote_code or True

            init_config = LLMInitConfig(
                name=local_model,
                loader=proj_cfg.loader,
                hf_model=proj_cfg.config.model_id_or_path,
                device=getattr(proj_cfg.config, "device", None),
                dtype=getattr(proj_cfg.config, "torch_dtype", None),
                trust_remote_code=getattr(proj_cfg.config, "trust_remote_code", True),
                api_model=None,
            )
            logger.info(f"[LLMManager] ✅ Using config: {init_config}")

            # Get Hugging Face token for gated models
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                logger.warning("No HF_TOKEN environment variable found. Some gated models may require authentication.")

            if init_config.loader == "transformers":
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except ImportError as e:
                    raise ImportError("transformers not installed; `pip install transformers`") from e

                try:
                    device = init_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
                    tokenizer = AutoTokenizer.from_pretrained(
                        init_config.hf_model,
                        token=hf_token,
                        use_safetensors=proj_cfg.config.use_safetensors,
                        trust_remote_code=init_config.trust_remote_code,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        init_config.hf_model,
                        token=hf_token,
                        torch_dtype=proj_cfg.config.resolved_dtype(),
                        device_map=init_config.device or "auto",
                        use_safetensors=proj_cfg.config.use_safetensors,
                        trust_remote_code=init_config.trust_remote_code,
                        low_cpu_mem_usage=proj_cfg.config.low_cpu_mem_usage,
                        offload_folder=proj_cfg.config.offload_folder,
                    )
                    self._client = (tokenizer, model)
                    logger.info(f"[LLMManager] ✅ Transformers model loaded: {local_model}")
                except Exception as e:
                    logger.error(f"[LLMManager] Failed to load transformers model {local_model}: {e}")
                    raise

            elif init_config.loader == "gptq":
                try:
                    from auto_gptq import AutoGPTQForCausalLM
                    from transformers import AutoTokenizer
                except ImportError as e:
                    raise ImportError("auto_gptq or transformers not installed; `pip install auto_gptq transformers`") from e

                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        proj_cfg.config.tokenizer_model_id or init_config.hf_model,
                        token=hf_token,
                        use_safetensors=proj_cfg.config.use_safetensors,
                        trust_remote_code=init_config.trust_remote_code,
                    )
                    model = AutoGPTQForCausalLM.from_quantized(
                        init_config.hf_model,
                        model_basename=proj_cfg.config.model_basename,
                        device=init_config.device or "cuda",
                        use_safetensors=proj_cfg.config.use_safetensors,
                        trust_remote_code=init_config.trust_remote_code,
                        disable_exllama=proj_cfg.config.disable_exllama,
                        group_size=proj_cfg.config.group_size,
                    )
                    self._client = (tokenizer, model)
                    logger.info(f"[LLMManager] ✅ GPTQ model loaded: {local_model}")
                except Exception as e:
                    logger.error(f"[LLMManager] Failed to load GPTQ model {local_model}: {e}")
                    raise

            elif init_config.loader == "awq":
                try:
                    from awq import AutoAWQForCausalLM
                    from transformers import AutoTokenizer
                except ImportError as e:
                    raise ImportError("awq or transformers not installed; `pip install autoawq transformers`") from e

                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        init_config.hf_model,
                        token=hf_token,
                        use_safetensors=proj_cfg.config.use_safetensors,
                        trust_remote_code=init_config.trust_remote_code,
                    )
                    model = AutoAWQForCausalLM.from_quantized(
                        init_config.hf_model,
                        device=init_config.device or "cuda",
                        use_safetensors=proj_cfg.config.use_safetensors,
                        trust_remote_code=init_config.trust_remote_code,
                        fuse_qkv=proj_cfg.config.fuse_qkv,
                    )
                    self._client = (tokenizer, model)
                    logger.info(f"[LLMManager] ✅ AWQ model loaded: {local_model}")
                except Exception as e:
                    logger.error(f"[LLMManager] Failed to load AWQ model {local_model}: {e}")
                    raise

            elif init_config.loader == "llama_cpp":
                try:
                    from llama_cpp import Llama
                except ImportError as e:
                    raise ImportError("llama_cpp_python not installed; `pip install llama_cpp_python`") from e
                try:
                    from loaders.model_configs_models import LlamaCppLoaderConfig
                    logger.debug(f"Checking config type for {local_model}: {type(proj_cfg.config).__name__}")
                    if not isinstance(proj_cfg.config, LlamaCppLoaderConfig):
                        logger.error(f"Expected LlamaCppLoaderConfig for {local_model}, got {type(proj_cfg.config).__name__}")
                        raise ValueError(f"Invalid config type for llama_cpp loader: {type(proj_cfg.config).__name__}")
                    n_ctx = proj_cfg.config.n_ctx or 2048
                    n_gpu_layers = proj_cfg.config.n_gpu_layers
                    model_path = proj_cfg.config.model_id_or_path
                    logger.info(f"[LLMManager] Loading Llama model from {model_path} with n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")
                    self._client = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        verbose=proj_cfg.config.verbose,
                    )
                    logger.info(f"[LLMManager] ✅ Llama model loaded: {local_model}")
                except Exception as e:
                    logger.error(f"[LLMManager] Failed to load Llama model {local_model}: {e}")
                    raise
            else:
                raise ValueError(f"Unsupported loader: {init_config.loader}")

        else:
            validate_api_model_name(api_model)
            self._client = OpenAIAdapter(model=api_model)
            logger.info(f"[LLMManager] 📦 API client ready for model: {api_model}")

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the LLM (local or API).

        Args:
            messages: List of {"role": str, "content": str} dicts.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters (e.g., max_tokens).

        Returns:
            str: The generated response.
        """
        if self._client is None:
            raise ValueError("LLM client not initialized.")

        if self.api_model:
            return await self._client.generate_chat_async(
                messages=messages,
                temperature=temperature,
                **kwargs,
            )

        if self.local_model:
            if isinstance(self._client, tuple):  # transformers, gptq, awq
                tokenizer, model = self._client
                # Fallback chat template for Llama-2 models
                chat_template = None
                if self.model_name == "llama_2_7b_chat_gptq":
                    chat_template = (
                        "{% for message in messages %}"
                        "{% if message['role'] == 'user' %}"
                        "[INST] {{ message['content'] }} [/INST]"
                        "{% elif message['role'] == 'system' %}"
                        "{{ message['content'] }} "
                        "{% elif message['role'] == 'assistant' %}"
                        "{{ message['content'] }} "
                        "{% endif %}"
                        "{% endfor %}"
                        "{{ '<|im_end|> <|im_start|> assistant' if add_generation_prompt else '' }}"
                    )
                try:
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        chat_template=chat_template,
                    )
                except Exception as e:
                    logger.error(f"Failed to apply chat template: {e}")
                    raise
                device = getattr(model, "device", "cuda")
                inputs = inputs.to(device)
                outputs = model.generate(
                    inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 512),
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    min_p=kwargs.get("min_p", 0.15),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.05),
                    top_p=kwargs.get("top_p", 0.9),
                )
                return tokenizer.decode(outputs[0], skip_special_tokens=True)

            elif isinstance(self._client, object):  # llama_cpp
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                response = self._client(
                    prompt=prompt,
                    max_tokens=kwargs.get("max_tokens", 512),
                    temperature=temperature,
                    top_p=kwargs.get("top_p", 0.9),
                )
                return response.get("choices", [{}])[0].get("text", "").strip()

        raise ValueError("No valid LLM client configured.")
