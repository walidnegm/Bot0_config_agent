# agent/model_config_models.py

from pathlib import Path
import torch
from typing import Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator


# 🔁 Literal for known loader types
ModelLoaderType = Literal["gguf", "gptq", "awq", "transformers"]


# 🎯 Base config for all HF-style models (GPTQ, AWQ, Transformers)
class BaseHFModelConfig(BaseModel):
    model_id: (
        str  # * In transformer based models, model_id can be either id or file path
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch_dtype: str = "float16"
    use_safetensors: Optional[bool] = None  # transformers only

    @field_validator("torch_dtype")
    @classmethod
    def validate_torch_dtype(cls, v: str) -> str:
        if not hasattr(torch, v):
            raise ValueError(f"Invalid torch dtype: '{v}'")
        return v

    def resolved_dtype(self) -> torch.dtype:
        return getattr(torch, self.torch_dtype)


# 📦 GGUF model config for llama.cpp
class GGUFModelConfig(BaseModel):
    """
    GGUF - specific to llama models
    """

    model_path: str  # * In llama_cpp and gguf, model_path has to be a file path
    n_ctx: int = 4096
    n_gpu_layers: int = Field(
        default_factory=lambda: -1 if torch.cuda.is_available() else 0
    )
    chat_format: str = "zephyr"
    verbose: bool = False

    @field_validator("model_path")
    @classmethod
    def resolve_and_validate_path(cls, v: str) -> str:
        path = Path(v).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Model path does not exist or is not a file: {path}")
        return str(path)


# 🧠 Top-level entry loaded from models_configs.json
class ModelConfigEntry(BaseModel):
    loader: ModelLoaderType
    config: Union[BaseHFModelConfig, GGUFModelConfig]

    @classmethod
    def parse_with_loader(cls, name: str, data: Dict[str, Any]) -> "ModelConfigEntry":
        if "loader" not in data or "config" not in data:
            raise ValueError(f"Missing loader or config block for model: {name}")

        loader = data["loader"]
        cfg = data["config"]

        if loader == "gguf":
            config = GGUFModelConfig(**cfg)
        elif loader in {"gptq", "awq", "transformers"}:
            config = BaseHFModelConfig(**cfg)
        else:
            raise ValueError(f"Unsupported loader: {loader}")

        return cls(loader=loader, config=config)
