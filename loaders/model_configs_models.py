# loaders/model_configs_models.py
from __future__ import annotations
from pathlib import Path
import torch
from typing import Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator
from huggingface_hub import snapshot_download
import os
import logging
import re

logger = logging.getLogger(__name__)

# 🔁 Literal for known loader types
ModelLoaderType = Literal["llama_cpp", "gptq", "awq", "transformers"]

class AWQLoaderConfig(BaseModel):
    model_id_or_path: str
    device: str = "auto"
    torch_dtype: str = "float16"
    use_safetensors: bool = True
    fuse_qkv: Optional[bool] = None

class GPTQLoaderConfig(BaseModel):
    model_id_or_path: str
    device: str = "auto"
    torch_dtype: str = "float16"
    disable_exllama: Optional[bool] = None
    group_size: Optional[int] = None
    use_safetensors: Optional[bool] = None
    trust_remote_code: Optional[bool] = None

class LlamaCppLoaderConfig(BaseModel):
    """
    model_id_or_path can be:
      - A local absolute/relative path to a .gguf file (preferred for reliability)
      - A Hugging Face repo id in the form 'org/repo' (requires gguf_file OR a single .gguf in the repo)
    """
    model_id_or_path: str
    gguf_file: Optional[str] = None  # e.g., "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = Field(default_factory=lambda: -1 if torch.cuda.is_available() else 0)
    chat_format: str = "zephyr"
    verbose: bool = False

    @field_validator("model_id_or_path")
    @classmethod
    def resolve_and_validate_path(cls, v: str, info) -> str:
        gguf_file = info.data.get("gguf_file")
        s = str(v).strip()
        p = Path(s).expanduser()

        # ---------- Prefer LOCAL PATH heuristics first ----------
        # Absolute path, or explicit relative path, or existing file with .gguf
        is_explicit_local = (
            p.is_absolute()
            or s.startswith("./")
            or s.startswith("../")
            or s.startswith("~/")
            or (p.suffix.lower() == ".gguf" and p.exists())
        )
        if is_explicit_local:
            # Resolve relative to project root if not absolute
            if not p.is_absolute():
                try:
                    from utils.find_root_dir import find_project_root
                    p = (find_project_root() / p).resolve()
                except Exception:
                    p = p.resolve()

            if not p.is_file():
                raise ValueError(f"Model path does not exist or is not a file: {p}")
            if p.suffix.lower() != ".gguf":
                logger.warning(f"[llama_cpp] Local model file does not have .gguf extension: {p.name}")
            logger.debug(f"[llama_cpp] Using local GGUF path: {p}")
            return str(p)

        # ---------- HF repo id branch (strict pattern: exactly one slash, not starting with '/') ----------
        is_repo_id = bool(re.match(r"^[^/]+/[^/]+$", s))
        if is_repo_id:
            repo_id = s
            allow_patterns = [f"*{gguf_file}"] if gguf_file else None
            if gguf_file:
                logger.debug(f"[llama_cpp] Repo '{repo_id}' with explicit gguf_file='{gguf_file}'")
            else:
                logger.debug(f"[llama_cpp] Repo '{repo_id}' with NO gguf_file; will list .gguf candidates")

            try:
                cache_dir = snapshot_download(
                    repo_id=repo_id,
                    token=os.getenv("HF_TOKEN"),
                    allow_patterns=allow_patterns,
                    ignore_patterns=["*.gitattributes", "*.md"],
                )
            except Exception as e:
                logger.error(f"[llama_cpp] Failed to snapshot_download repo '{repo_id}': {e}")
                raise ValueError(f"Failed to download/resolve Hugging Face repo '{repo_id}': {e}")

            cache_dir = Path(cache_dir)
            gguf_candidates = list(cache_dir.rglob("*.gguf"))

            if gguf_file:
                matches = [p for p in gguf_candidates if p.name == gguf_file]
                if not matches:
                    sample = [p.name for p in gguf_candidates][:10]
                    raise ValueError(
                        f"GGUF file '{gguf_file}' not found in repo '{repo_id}'. "
                        f"Found candidates: {sample}"
                    )
                return str(matches[0].resolve())

            # No gguf_file specified → require exactly one candidate
            if len(gguf_candidates) == 0:
                raise ValueError(
                    f"No .gguf files found in repo '{repo_id}'. "
                    f"Please set 'gguf_file' under config."
                )
            if len(gguf_candidates) > 1:
                names = sorted({p.name for p in gguf_candidates})
                raise ValueError(
                    "Multiple .gguf files were found in the repo. "
                    "Please add 'gguf_file' under config to choose one of:\n  - " + "\n  - ".join(names)
                )
            logger.info(f"[llama_cpp] Auto-selected the only GGUF in repo '{repo_id}': {gguf_candidates[0].name}")
            return str(gguf_candidates[0].resolve())

        # ---------- Fallback: try to treat as local file path ----------
        if not p.is_absolute():
            try:
                from utils.find_root_dir import find_project_root
                p = (find_project_root() / p).resolve()
            except Exception:
                p = p.resolve()

        if not p.is_file():
            raise ValueError(
                f"Could not interpret '{s}' as a local .gguf file or a valid HF repo id ('org/repo'). "
                f"Path resolved to '{p}', which is not a file."
            )
        return str(p)

class TransformersLoaderConfig(BaseModel):
    model_id_or_path: str
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch_dtype: str = "float16"
    use_safetensors: Optional[bool] = None
    trust_remote_code: Optional[bool] = None
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

class LoaderConfigEntry(BaseModel):
    loader: ModelLoaderType
    config: Union[
        LlamaCppLoaderConfig,
        TransformersLoaderConfig,
        GPTQLoaderConfig,
        AWQLoaderConfig,
    ]
    generation_config: Optional[Dict[str, Any]] = None

    @classmethod
    def parse_with_loader(cls, name: str, data: Dict[str, Any]) -> LoaderConfigEntry:
        if "loader" not in data or "config" not in data:
            logger.error(f"Missing loader or config block for model: {name}")
            raise ValueError(f"Missing loader or config block for model: {name}")

        loader = data["loader"]
        cfg = data["config"]
        gen_cfg = data.get("generation_config")

        logger.debug(f"Parsing config for model '{name}' with loader '{loader}', config: {cfg}")
        try:
            if loader == "llama_cpp":
                config = LlamaCppLoaderConfig(**cfg)
            elif loader == "gptq":
                config = GPTQLoaderConfig(**cfg)
            elif loader == "awq":
                config = AWQLoaderConfig(**cfg)
            elif loader == "transformers":
                config = TransformersLoaderConfig(**cfg)
            else:
                logger.error(f"Unsupported loader: {loader}")
                raise ValueError(f"Unsupported loader: {loader}")
        except Exception as e:
            logger.error(f"Failed to validate config for model '{name}' with loader '{loader}': {e}")
            raise
        logger.debug(f"Validated config for model '{name}': config_type={type(config).__name__}")
        return cls(loader=loader, config=config, generation_config=gen_cfg)
