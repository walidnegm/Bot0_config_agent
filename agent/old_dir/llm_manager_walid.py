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
                logger.debug(f"Loaded config for '{local_model}': {proj_cfg}")
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Failed to load model configuration for '{local_model}' from {MODEL_CONFIGS_YAML_FILE}: {e}")
                raise

            init_config = LLMInitConfig(
                name=local_model,
                loader=proj_cfg.loader,
                hf_model=proj_cfg.config.model_id_or_path,
                device=proj_cfg.config.device,
                dtype=proj_cfg.config.torch_dtype,
                trust_remote_code=proj_cfg.config.trust_remote_code or True,
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
                    self._client = (
                        AutoTokenizer.from_pretrained(
                            init_config.hf_model,
                            use_fast=True,
                            trust_remote_code=init_config.trust_remote_code,
                            use_safetensors=proj_cfg.config.use_safetensors,
                            token=hf_token,
                            local_files_only=True,
                        ),
                        AutoModelForCausalLM.from_pretrained(
                            init_config.hf_model,
                            device_map="auto",
                            trust_remote_code=init_config.trust_remote_code,
                            torch_dtype=init_config.dtype or "auto",
                            use_safetensors=proj_cfg.config.use_safetensors,
                            low_cpu_mem_usage=proj_cfg.config.low_cpu_mem_usage or False,
                            offload_folder=proj_cfg.config.offload_folder,
                            token=hf_token,
                            local_files_only=True,
                        ),
                    )
                    logger.info(f"[LLMManager] 📦 Local model loaded: {init_config.name} on {device}")
                except Exception as e:
                    logger.error(f"[LLMManager] Failed to load local model {init_config.name}: {e}")
                    if "401 Client Error" in str(e):
                        logger.error(
                            f"Authentication failed for {init_config.hf_model}. "
                            f"Ensure the model is downloaded locally to $HF_HOME/hub/models--{init_config.hf_model.replace('/', '--')}/snapshots or you have access via HF_TOKEN."
                        )
                    raise

            elif init_config.loader == "llama_cpp":
                try:
                    from llama_cpp import Llama
                except ImportError as e:
                    raise ImportError("llama_cpp_python not installed; `pip install llama_cpp_python`") from e

                try:
                    cache_dir = os.path.join(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
                    model_cache_path = os.path.join(cache_dir, f"models--{init_config.hf_model.replace('/', '--')}")
                    snapshots_dir = os.path.join(model_cache_path, "snapshots")
                    if not os.path.exists(snapshots_dir):
                        logger.error(f"No snapshots found for model {init_config.hf_model} at {snapshots_dir}")
                        raise ValueError(f"No snapshots found for model {init_config.hf_model}")
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if not snapshots:
                        logger.error(f"No snapshot directories found for model {init_config.hf_model} at {snapshots_dir}")
                        raise ValueError(f"No snapshot directories found for model {init_config.hf_model}")
                    latest_snapshot = max(snapshots, key=lambda d: os.path.getmtime(os.path.join(snapshots_dir, d)))
                    model_path = os.path.join(snapshots_dir, latest_snapshot, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
                    if not os.path.isfile(model_path):
                        logger.error(f"Local model file does not exist: {model_path}")
                        raise ValueError(f"Local model file does not exist: {model_path}")
                    self._client = Llama(
                        model_path=str(model_path),
                        n_ctx=proj_cfg.config.n_ctx or 2048,
                        n_gpu_layers=proj_cfg.config.n_gpu_layers or 0,
                        verbose=proj_cfg.config.verbose or False,
                    )
                    logger.info(f"[LLMManager] 📦 Local Llama model loaded: {init_config.name}")
                except Exception as e:
                    logger.error(f"[LLMManager] Failed to load Llama model {init_config.name}: {e}")
                    raise

            elif init_config.loader == "gptq":
                try:
                    from gptqmodel import GPTQModel
                    from transformers import AutoTokenizer
                except ImportError as e:
                    raise ImportError("gptqmodel or transformers not installed; `pip install gptqmodel transformers`") from e

                try:
                    self._client = (
                        AutoTokenizer.from_pretrained(
                            init_config.hf_model,
                            use_fast=True,
                            token=hf_token,
                            local_files_only=True,
                        ),
                        GPTQModel.from_quantized(
                            init_config.hf_model,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            token=hf_token,
                        ),
                    )
                    logger.info(f"[LLMManager] 📦 Local GPTQ model loaded: {init_config.name}")
                except Exception as e:
                    logger.error(f"[LLMManager] Failed to load GPTQ model {init_config.name}: {e}")
                    if "401 Client Error" in str(e):
                        logger.error(
                            f"Authentication failed for {init_config.hf_model}. "
                            f"Ensure the model is downloaded locally to $HF_HOME/hub/models--{init_config.hf_model.replace('/', '--')}/snapshots or you have access via HF_TOKEN."
                        )
                    raise

            elif init_config.loader == "awq":
                try:
                    from awq import AutoAWQForCausalLM
                    from transformers import AutoTokenizer
                except ImportError as e:
                    raise ImportError("autoawq or transformers not installed; `pip install autoawq transformers`") from e

                try:
                    self._client = (
                        AutoTokenizer.from_pretrained(
                            init_config.hf_model,
                            use_fast=True,
                            token=hf_token,
                            local_files_only=True,
                        ),
                        AutoAWQForCausalLM.from_quantized(
                            init_config.hf_model,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            fuse_layers=False,
                            token=hf_token,
                        ),
                    )
                    logger.info(f"[LLMManager] 📦 Local AWQ model loaded: {init_config.name}")
                except Exception as e:
                    logger.error(f"[LLMManager] Failed to load AWQ model {init_config.name}: {e}")
                    if "401 Client Error" in str(e):
                        logger.error(
                            f"Authentication failed for {init_config.hf_model}. "
                            f"Ensure the model is downloaded locally to $HF_HOME/hub/models--{init_config.hf_model.replace('/', '--')}/snapshots or you have access via HF_TOKEN."
                        )
                    raise

            else:
                raise ValueError(f"Unsupported loader: {init_config.loader}")

        else:  # api_model
            validate_api_model_name(api_model)
            provider = get_llm_provider(api_model)
            logger.info(f"[LLMManager] Using API provider: {provider} for model: {api_model}")
            init_config = LLMInitConfig(
                name=api_model,
                loader="openai",
                api_model=api_model,
            )
            logger.info(f"[LLMManager] 📦 API client ready for model: {api_model}")
            self._client = OpenAIAdapter(model=api_model)

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
