# agent/model_config_models.py
from __future__ import annotations
from pathlib import Path
import torch
from typing import Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator


# 🔁 Literal for known loader types
ModelLoaderType = Literal["llama_cpp", "gptq", "awq", "transformers"]


class AWQLoaderConfig(BaseModel):
    model_id_or_path: str
    device: str = "auto"
    torch_dtype: str = "float16"
    use_safetensors: bool = True
    fuse_qkv: Optional[bool] = None  # example AWQ-specific flag


class GPTQLoaderConfig(BaseModel):
    model_id_or_path: str
    device: str = "auto"
    torch_dtype: str = "float16"
    disable_exllama: Optional[bool] = None  # example GPTQ-specific
    group_size: Optional[int] = None


# 📦 GGUF model config for llama.cpp
class LlamaCppLoaderConfig(BaseModel):
    model_id_or_path: str  # * In llama_cpp and gguf, model_path has to be a file path
    n_ctx: int = 4096
    n_gpu_layers: int = Field(
        default_factory=lambda: -1 if torch.cuda.is_available() else 0
    )
    chat_format: str = "zephyr"
    verbose: bool = False

    @field_validator("model_id_or_path")
    @classmethod
    def resolve_and_validate_path(cls, v: str) -> str:
        path = Path(v).expanduser()

        # If not absolute, assume it's relative to project root
        if not path.is_absolute():
            from utils.find_root_dir import find_project_root

            path = find_project_root() / path

        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Model path does not exist or is not a file: {path}")

        return str(path)


# 🎯 Base config for all HF-style models (GPTQ, AWQ, Transformers)
class TransformersLoaderConfig(BaseModel):
    model_id_or_path: (
        str  # * In transformer based models, model_id can be either id or file path
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch_dtype: str = "float16"
    use_safetensors: Optional[bool] = None  # transformers only
    trust_remote_code: Optional[bool] = False
    low_cpu_mem_usage: Optional[bool] = None
    offload_folder: Optional[str] = None

    @field_validator("torch_dtype")
    @classmethod
    def validate_torch_dtype(cls, v: str) -> str:
        if not hasattr(torch, v):
            raise ValueError(f"Invalid torch dtype: '{v}'")
        return v

    def resolved_dtype(self) -> torch.dtype:
        return getattr(torch, self.torch_dtype)


class AWQLoaderEntry(BaseModel):
    loader: Literal["awq"]
    config: AWQLoaderConfig


class GPTQLoaderEntry(BaseModel):
    loader: Literal["gptq"]
    config: GPTQLoaderConfig


class LlamaCppLoaderEntry(BaseModel):
    loader: Literal["llama_cpp"]
    config: LlamaCppLoaderConfig


class TransformersLoaderEntry(BaseModel):
    loader: Literal["transformers"]
    config: TransformersLoaderConfig


# 🧠 Top-level entry loaded from models_configs.json
class LoaderConfigEntry(BaseModel):
    """
    Represents a validated model configuration entry, including the loader type
    and its corresponding configuration.

    This is the top-level structure used when parsing a single model entry
    from `models_configs.yaml` or `models_configs.json`.

    The `loader` field determines which backend will be used to load and run
    the model (e.g., 'llama_cpp', 'gptq', 'awq', 'transformers'), and the
    `config` field holds the model-specific settings validated against the
    appropriate Pydantic model.

    Fields:
        loader (ModelLoaderType):
            A literal string indicating which runtime should be used to load the model.
            Supported values: "llama_cpp", "gptq", "awq", "transformers".

        config (Union[...]):
            The model configuration, validated against one of the loader-specific
            Pydantic models:
            - LlamaCppLoaderConfig
            - GPTQLoaderConfig
            - AWQLoaderConfig
            - TransformersLoaderConfig

    Methods:
        parse_with_loader(name: str, data: dict) -> LoaderConfigEntry:
            Factory method to parse and validate a model entry based on
                the loader type.
            Automatically dispatches the correct Pydantic config schema for
                the given loader.
    """

    loader: ModelLoaderType
    config: Union[
        TransformersLoaderConfig,
        LlamaCppLoaderConfig,
        GPTQLoaderConfig,
        AWQLoaderConfig,
    ]

    @classmethod
    def parse_with_loader(cls, name: str, data: Dict[str, Any]) -> LoaderConfigEntry:
        """
        Parse and validate a model config entry using the appropriate config class
        based on the loader type.

        Args:
            name (str): Name of the model (used for error messages).
            data (dict): Dictionary containing 'loader' and 'config' keys.

        Returns:
            LoaderConfigEntry: Fully validated model entry.

        Raises:
            ValueError: If required fields are missing or loader type is unsupported.
        """

        if "loader" not in data or "config" not in data:
            raise ValueError(f"Missing loader or config block for model: {name}")

        loader = data["loader"]
        cfg = data["config"]

        if loader == "llama_cpp":
            config = LlamaCppLoaderConfig(**cfg)
        elif loader == "gptq":
            config = GPTQLoaderConfig(**cfg)
        elif loader == "awq":
            config = AWQLoaderConfig(**cfg)
        elif loader == "transformers":
            config = TransformersLoaderConfig(**cfg)
        else:
            raise ValueError(f"Unsupported loader: {loader}")

        return cls(loader=loader, config=config)
