"""agent/model_config_models.py"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Union, Dict, Any
import torch
from pydantic import BaseModel, Field, field_validator
from bot0_config_agent.utils.system.find_root_dir import find_project_root

# ------------------------------------------------------------
# Loader type enum
# ------------------------------------------------------------
ModelLoaderType = Literal["llama_cpp", "gptq", "awq", "transformers"]


# ------------------------------------------------------------
# Common / helper models (HF-style)
# ------------------------------------------------------------
class HFLoadConfig(BaseModel):
    """
    Loader-agnostic HF load kwargs. These map 1:1 to kwargs passed to
    AutoModelForCausalLM.from_pretrained(...) or quant loaders (after filtering).
    """

    model_id_or_path: str
    device_map: Optional[Union[str, Dict[str, Any]]] = None
    max_memory: Optional[Dict[Union[int, str], str]] = None
    offload_folder: Optional[str] = None
    torch_dtype: Optional[str] = None
    trust_remote_code: Optional[bool] = False
    low_cpu_mem_usage: Optional[bool] = None

    @field_validator("torch_dtype")
    @classmethod
    def _validate_torch_dtype_str(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        s = str(v)
        if not hasattr(torch, s):
            raise ValueError(f"Invalid torch dtype: '{s}'")
        return s


class AWQQuantConfig(BaseModel):
    """
    AWQ-specific quantization options. Keep minimal; extend as needed.
    """

    fuse_layers: Optional[bool] = None
    trust_remote_code: Optional[bool] = None


class GPTQQuantConfig(BaseModel):
    """
    GPTQ-specific quantization options. Keep minimal; extend as needed.
    """

    bits: Optional[int] = None
    group_size: Optional[int] = None
    desc_act: Optional[bool] = None
    trust_remote_code: Optional[bool] = None


# ------------------------------------------------------------
# Loader-specific config models
# ------------------------------------------------------------
class TransformersLoaderConfig(BaseModel):
    """
    Standard HF transformers loader config.
    """

    load_config: HFLoadConfig
    # transformers usually has no quant_config; keep for forward-compat
    quant_config: Optional[Dict[str, Any]] = None


class AWQLoaderConfig(BaseModel):
    """
    AutoAWQ loader config with split load/quant sections.
    """

    load_config: HFLoadConfig
    quant_config: Optional[AWQQuantConfig] = None


class GPTQLoaderConfig(BaseModel):
    """
    GPTQ loader config with split load/quant sections.
    """

    load_config: HFLoadConfig
    quant_config: Optional[GPTQQuantConfig] = None


class LlamaCppLoaderConfig(BaseModel):
    """
    Llama.cpp / GGUF loader config.

    Design:
      • Keep YAML as the single source of truth. All llama.cpp kwargs live under
        `load_config` exactly as written in YAML (e.g., `n_ctx`, `n_gpu_layers`,
        `chat_format`, `verbose`, etc.). We do not normalize or reshape these.
      • The only required key is `load_config.model_id_or_path`, which must point
        to a local `.gguf` file for llama.cpp. We resolve it to an absolute path
        at validation time so downstream code can pass it directly.
      • The runtime loader will simply do:
            kwargs = dict(config.load_config)
            kwargs["model_path"] = kwargs.pop("model_id_or_path")
            self.model = Llama(**kwargs)
        preserving your YAML’s keyword structure.

    Expected YAML snippet:
        tinyllama_1_1b_chat_gguf:
          loader: llama_cpp
          load_config:
            model_id_or_path: models/tinyllama/tinyllama.Q4_K_M.gguf
            n_ctx: 4096
            n_gpu_layers: -1
            chat_format: zephyr
            verbose: true
          generation_config:
            max_tokens: 512
            temperature: 0.3
            top_p: 0.95
    """

    load_config: Dict[str, Any] = Field(default_factory=Dict)
    # quant_config intentionally NOT used for llama.cpp

    @field_validator("load_config")
    @classmethod
    def validate_load_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate: model_id_or_path must be a valid directory path under the root project
        """
        model_path = v.get("model_id_or_path")
        if not model_path:
            raise ValueError("load_config.model_id_or_path is required for llama_cpp")

        p = Path(model_path).expanduser()
        if not p.is_absolute():
            p = (find_project_root() / p).resolve()
        if not p.is_file():
            raise ValueError(f"Model path does not exist or is not a file: {p}")

        v["model_id_or_path"] = str(p)
        return v


# ------------------------------------------------------------
# Top-level normalized entry
# ------------------------------------------------------------
class LoaderConfigEntry(BaseModel):
    """
    Neutral top-level model configuration entry:
      - loader: which backend to use
      - config: loader-specific, validated model (above)
      - generation_config: inference kwargs (temperature, max_new_tokens, ...)
    """

    loader: ModelLoaderType
    config: Union[
        TransformersLoaderConfig,
        LlamaCppLoaderConfig,
        GPTQLoaderConfig,
        AWQLoaderConfig,
    ]
    generation_config: Optional[Dict[str, Any]] = None

    @classmethod
    def parse_with_loader(cls, name: str, data: Dict[str, Any]) -> "LoaderConfigEntry":
        """
        Accepts any of the following shapes and normalizes them:

        a) Legacy:
           {
             "loader": "...",
             "config": { ... loader-specific ... },
             "generation_config": {...}
           }

        b) New format (top-level blocks):
           {
             "loader": "...",
             "load_config": {...},      # required for HF loaders
             "quant_config": {...},     # optional (AWQ/GPTQ)
             "generation_config": {...}
           }

        c) Legacy-flat HF config (rare):
           {
             "loader": "transformers" | "awq" | "gptq",
             "config": { "model_id_or_path": "...", "torch_dtype": "...", ... }
           }
           (We auto-wrap that into {"load_config": {...}}.)
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"Model entry for {name} must be a mapping, got {type(data)}"
            )

        # --- Normalize new-format top-level keys into a 'config' dict ---
        if "config" not in data and ("load_config" in data or "quant_config" in data):
            data = dict(data)  # shallow copy
            cfg: Dict[str, Any] = {}
            if "load_config" in data:
                cfg["load_config"] = data.pop("load_config")
            if "quant_config" in data:
                cfg["quant_config"] = data.pop("quant_config")
            data["config"] = cfg

        if "loader" not in data or "config" not in data:
            raise ValueError(f"Missing loader or config block for model: {name}")

        loader = data["loader"]
        cfg = data["config"] or {}
        gen_cfg = data.get("generation_config")

        # --- Auto-wrap legacy flat dicts for HF loaders into load_config ---
        if loader in ("transformers", "awq", "gptq"):
            if "load_config" not in cfg and (
                "model_id_or_path" in cfg
                or "device_map" in cfg
                or "torch_dtype" in cfg
                or "trust_remote_code" in cfg
            ):
                cfg = {"load_config": cfg}

        # --- Dispatch to concrete config models ---
        if loader == "transformers":
            config_obj = TransformersLoaderConfig(**cfg)
        elif loader == "awq":
            config_obj = AWQLoaderConfig(**cfg)
        elif loader == "gptq":
            config_obj = GPTQLoaderConfig(**cfg)
        elif loader == "llama_cpp":
            # remains flat
            config_obj = LlamaCppLoaderConfig(**cfg)
        else:
            raise ValueError(f"Unsupported loader: {loader}")

        return cls(loader=loader, config=config_obj, generation_config=gen_cfg)


__all__ = [
    "ModelLoaderType",
    "HFLoadConfig",
    "AWQQuantConfig",
    "GPTQQuantConfig",
    "TransformersLoaderConfig",
    "AWQLoaderConfig",
    "GPTQLoaderConfig",
    "LlamaCppLoaderConfig",
    "LoaderConfigEntry",
]
