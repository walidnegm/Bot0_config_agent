"""agent/model_config_models.py"""

from __future__ import annotations
from pathlib import Path
import re
import logging
from typing import Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, AliasChoices

# from bot0_config_agent.utils.system.find_root_dir import find_project_root
from bot0_config_agent.configs.paths import ROOT_DIR

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Loader type enum
# ------------------------------------------------------------
ModelLoaderType = Literal["llama_cpp", "gptq", "awq", "transformers"]
QuantizationMethod = Literal["AWQ", "FP", "GGUF", "GPTQ"]

# Hugging Face Repository Regex
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


# ------------------------------------------------------------
# Common / helper models (HF-style)
# ------------------------------------------------------------
class HFLoadConfig(BaseModel):
    """
    Loader-agnostic HF load kwargs. These map 1:1 to kwargs passed to
    AutoModelForCausalLM.from_pretrained(...) or quant loaders (after filtering).
    """

    model_id_or_path: str
    device_map: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        validation_alias=AliasChoices(
            "device", "device_map"
        ),  # accept `device:` from YAML
    )
    use_safetensors: Optional[bool] = None
    max_memory: Optional[Dict[Union[int, str], str]] = None
    offload_folder: Optional[str] = None
    torch_dtype: Optional[str] = None  # Keep as str here in the model
    trust_remote_code: Optional[bool] = False
    low_cpu_mem_usage: Optional[bool] = None

    # Derived, not from YAML:
    source: Literal["local_path", "hf_repo"] = "hf_repo"

    # local files only
    local_files_only: Optional[bool] = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("model_id_or_path")
    @classmethod
    def _normalize_model_id_or_path(cls, v: str, info):
        s = v.strip()
        # Local path?
        p = Path(s).expanduser()
        if p.is_absolute() or s.startswith(("./", "../", "~/")):
            p = p if p.is_absolute() else p.resolve()
            if not p.exists():
                raise ValueError(f"Local path does not exist: {p}")

            # mutate via return; source will be set in after validator below
            return str(p)

        # If it exists relative to project/root, you can try resolving there if desired:
        # from bot0_config_agent.utils.system.find_root_dir import find_project_root
        # pr = find_project_root()
        # if (pr / s).exists(): return str((pr / s).resolve())

        # Repo-like ID?
        if _HF_REPO_RE.match(s):
            return s

        # Ambiguous string: allow but warn (keeps backward compat)
        # You could instead raise to force clarity.
        return s

    @field_validator("source", mode="after")
    @classmethod
    def _set_source(cls, v, info):
        s = info.data.get("model_id_or_path", "") or ""
        p = Path(s).expanduser()
        if p.is_absolute() or p.exists():
            return "local_path"
        return "hf_repo"

    @field_validator("torch_dtype")
    @classmethod
    def _validate_torch_dtype_str(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v

        # accept torch.dtype objects or strings (with aliases)
        import torch

        # torch.dtype → canonical string
        if isinstance(v, torch.dtype):
            MAP = {
                torch.float16: "float16",
                torch.float32: "float32",
                torch.bfloat16: "bfloat16",
            }
            if v in MAP:
                return MAP[v]
            raise ValueError(f"Unsupported torch dtype: {v}")

        # string → canonical string
        s = str(v).strip().lower().replace("torch.", "")
        ALIASES = {
            "fp16": "float16",
            "half": "float16",
            "f16": "float16",
            "float16": "float16",
            "fp32": "float32",
            "float32": "float32",
            "float": "float32",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
        }
        if s in ALIASES:
            return ALIASES[s]

        # last chance: allow exact torch attribute names if valid
        if hasattr(torch, s) and getattr(torch, s) in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            return s

        raise ValueError(f"Invalid torch dtype: {v!r}")


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
    model_config = ConfigDict(frozen=True, extra="forbid")


class AWQLoaderConfig(BaseModel):
    """
    AutoAWQ loader config with split load/quant sections.
    """

    load_config: HFLoadConfig
    model_config = ConfigDict(frozen=True, extra="forbid")


class GPTQLoaderConfig(BaseModel):
    """
    GPTQ loader config with split load/quant sections.
    """

    load_config: HFLoadConfig
    model_config = ConfigDict(frozen=True, extra="forbid")


class LlamaCppLoaderConfig(BaseModel):
    """
    Llama.cpp / GGUF loader config.

    Design:
        - Keep YAML as the single source of truth. All llama.cpp kwargs live
            under `load_config` exactly as written in YAML (e.g., `n_ctx`,
            `n_gpu_layers`, `chat_format`, `verbose`, etc.). We do not normalize
            or reshape these.
        - The only required key is `load_config.model_id_or_path`, which must
            point to a local `.gguf` **file** for llama.cpp. We resolve it to
            an absolute path at validation time so downstream code can pass it
            directly.
        - The runtime loader will simply do:
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

    load_config: Dict[str, Any] = Field(default_factory=dict)
    # quant_config intentionally NOT used for llama.cpp

    @field_validator("load_config")
    @classmethod
    def validate_load_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate: model_id_or_path must be a valid file path under the root project
        """
        model_path = v.get("model_id_or_path")
        if not model_path:
            raise ValueError("load_config.model_id_or_path is required for llama_cpp")

        p = Path(model_path).expanduser()
        if not p.is_absolute():
            p = (ROOT_DIR / p).resolve()
        if not p.is_file():
            raise ValueError(f"Model path does not exist or is not a file: {p}")

        v["model_id_or_path"] = str(p)
        return v

    model_config = ConfigDict(frozen=True, extra="allow")


# ------------------------------------------------------------
# Top-level normalized entry
# ------------------------------------------------------------
_LOADCFG_BY_LOADER = {
    "transformers": TransformersLoaderConfig,
    "awq": AWQLoaderConfig,
    "gptq": GPTQLoaderConfig,
    "llama_cpp": LlamaCppLoaderConfig,
}


class LoaderConfigEntry(BaseModel):
    """
    Top-level normalized model configuration entry (new format only).

    Required explicit format:
        {
          "name": "<key>",                    # canonical key, e.g. "lfm2_1_2b"
          "loader": "<backend>",              # transformers | awq | gptq | llama_cpp
          "model_id": "<repo_or_path>",       # e.g. "LiquidAI/LFM2-1.2B"
          "model_description": "...",         # optional human-friendly text
          "quantization_method": "FP",        # optional: "GPTQ" | "AWQ" | "FP" | ...

          "load_config": {...},               # loader kwargs (HF/llama.cpp)
          "quant_config": {...},              # quantization options (AWQ/GPTQ only)
          "generation_config": {...}          # inference params (temperature, top_p, max_new_tokens)
        }

    Notes:
      - `generation_config` is strictly inference-time; do not mix with loader/quant configs.
      - This class does not auto-wrap any legacy shapes—YAML must match the schema above.
    """

    # --- Metadata ---
    name: str
    model_id: str
    model_description: Optional[str] = None
    quantization_method: Optional["QuantizationMethod"] = None

    # --- Loader backend ---
    loader: "ModelLoaderType"  # transformers | awq | gptq | llama_cpp

    # --- Config buckets ---
    load_config: Union[
        "TransformersLoaderConfig",
        "LlamaCppLoaderConfig",
        "GPTQLoaderConfig",
        "AWQLoaderConfig",
    ]
    quant_config: Dict[str, Any] = Field(default_factory=dict)
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def parse_entry(cls, name: str, data: Dict[str, Any]) -> "LoaderConfigEntry":
        """
        Build a validated LoaderConfigEntry from a raw YAML model entry.

        Workflow:
          1. Validate presence of required keys ("loader", "model_id", "load_config").
          2. Select the appropriate typed loader config class
             (Transformers, AWQ, GPTQ, or LlamaCpp) using `_LOADCFG_BY_LOADER`.
          3. Instantiate the loader-specific config model (`load_config`).
          4. Copy optional `quant_config` and `generation_config` blocks as plain dicts.
          5. Construct and return a fully normalized LoaderConfigEntry with
             metadata, typed load_config, and optional configs.
          6. Perform a non-fatal consistency check:
             log a warning if `model_id` does not match
             `load_config.model_id_or_path`.

        Args:
            name: Canonical model key (e.g. "lfm2_1_2b").
            data: Parsed YAML entry containing loader, model_id, configs.

        Returns:
            LoaderConfigEntry: fully typed, normalized configuration for one model.

        Raises:
            ValueError: If required keys are missing or the loader is unsupported.
            TypeError:  If `load_config` is not a dict.
        """
        # --- Required keys ---
        required = ("loader", "model_id", "load_config")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required key(s) {missing} for model '{name}'")

        loader = data["loader"]
        try:
            LoadCfgModel = _LOADCFG_BY_LOADER[loader]
        except KeyError:
            raise ValueError(f"Unsupported loader '{loader}' for model '{name}'")

        raw_load_cfg = data.get("load_config")
        if not isinstance(raw_load_cfg, dict):
            raise TypeError(
                f"'load_config' for model '{name}' must be a mapping, got {type(raw_load_cfg).__name__}"
            )

        # Typed loader-specific config
        load_cfg = LoadCfgModel(load_config=raw_load_cfg)

        # Optional buckets (leave as flexible dicts)
        quant_cfg = data.get("quant_config") or {}
        gen_cfg = data.get("generation_config") or {}

        entry = cls(
            name=name,
            loader=loader,
            model_id=data["model_id"],
            model_description=data.get("model_description"),
            quantization_method=data.get("quantization_method"),
            load_config=load_cfg,
            quant_config=dict(quant_cfg),
            generation_config=dict(gen_cfg),
        )

        # --- Consistency check: model_id vs load_config.model_id_or_path (non-fatal) ---
        mod_id_load = None
        if loader == "llama_cpp":
            # llama_cpp keeps a raw dict under load_config
            mod_id_load = entry.load_config.load_config.get("model_id_or_path")  # type: ignore[union-attr]
        else:
            # HF-style loaders have HFLoadConfig with a typed attribute
            mod_id_load = entry.load_config.load_config.model_id_or_path  # type: ignore[union-attr]

        if (
            isinstance(mod_id_load, str)
            and mod_id_load.strip()
            and mod_id_load != entry.model_id
        ):
            logger.warning(
                "[LoaderConfigEntry] model_id (%s) != load_config.model_id_or_path (%s) for '%s'",
                entry.model_id,
                mod_id_load,
                name,
            )

        return entry


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
