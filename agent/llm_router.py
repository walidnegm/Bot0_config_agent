"""
Module: agent/llm_rounter.py
"""

from pathlib import Path
from typing import Optional, Union
import logging
import yaml
from agent import llm_openai
from agent.llm_manager import (
    LLMManager,
)  # must support custom model_path, dtype, device_map
from utils.system.find_root_dir import find_project_root
from configs.paths import MODEL_CONFIGS_YAML_FILE

logger = logging.getLogger(__name__)
project_root = find_project_root()


def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)


class LLMRouter:
    """TBA"""

    def __init__(self, config_path: Path = MODEL_CONFIGS_YAML_FILE):
        self.config = self._load_config(config_path)
        self.models = self.config.get("models", {})
        self.default_model_key = self.config.get("default_model")
        self._llm_instances = {}  # cache of loaded local models

    def _load_config(self, path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"models.yaml config not found: {path}")
        with path.open("r") as f:
            return yaml.safe_load(f)

    def _get_model_cfg(self, model_key: str) -> dict:
        if model_key not in self.models:
            raise ValueError(f"Model key '{model_key}' not found in models.yaml.")
        return self.models[model_key]

    def _get_or_load_llm_manager(self, model_key: str, cfg: dict) -> LLMManager:
        if model_key in self._llm_instances:
            return self._llm_instances[model_key]

        print(
            f"[LLMRouter] 🔧 Loading local model '{model_key}' from {cfg['model_path']}"
        )

        # todo: need to update llmmaager
        manager = LLMManager(
            model_path=cfg["model_path"],
            quantization=cfg.get("quantization", "full"),
            dtype=cfg.get("dtype", "float16"),
            device_map=cfg.get("device_map", "auto"),
        )
        self._llm_instances[model_key] = manager
        return manager

    # Orchestrate generate method in LLMManager
    def generate(
        self,
        prompt: str,
        model_key: Optional[str],
        temperature: Union[float, str, None] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        model_key = model_key or self.default_model_key
        if not isinstance(model_key, str):
            raise ValueError("Model key must be a string and cannot be None.")
        cfg = self._get_model_cfg(model_key)

        # Coerce temperature to float if it's a string
        try:
            if temperature is not None and not isinstance(temperature, float):
                temperature = float(temperature)
        except ValueError:
            raise ValueError(
                f"Invalid temperature value: {temperature!r} (must be float-compatible)"
            )

        # Normalize temperature
        temp_min, temp_max = cfg.get("temp_range", [0.0, 1.0])
        temperature = clamp(
            temperature or cfg.get("default_temperature", 0.7), temp_min, temp_max
        )

        max_new_tokens = max_new_tokens or cfg.get("max_tokens", 512)
        system_prompt = system_prompt or cfg.get("system_prompt", "")

        if cfg["provider"] == "openai":
            print(f"[LLMRouter] ✈ Routing to OpenAI: {cfg['model']}")
            return llm_openai.generate(
                prompt, temperature=temperature, model=cfg["model"]
            )
        elif cfg["provider"] == "huggingface":
            manager = self._get_or_load_llm_manager(model_key, cfg)
            return manager.generate(
                user_prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unsupported provider '{cfg['provider']}' in models.yaml")
