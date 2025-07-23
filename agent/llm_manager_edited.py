"""
GGUF (General-purpose quantized format for efficient inference of large language models
on devices with limited resources, with suffix .gguf):
A binary format designed for efficient inference of transformer models, particularly large
language models.

GPTQ (General-Purpose Quantization for Transformers): A post-training quantization technique for
large language models, reducing model size and improving inference speed while maintaining accuracy.

AWQ (Activation-aware Weight Quantization): A quantization method that protects salient weights
based on activation distribution, preserving model quality while reducing model size and improving
inference efficiency.

* Changes to make:
* - add awq
* - use models.yaml
* - use the gptq load (use gptqmodel lib, not autoawq)

"""

# * Changes to make here:
import logging
from typing import Optional, Literal, Dict, Any, Union
from pathlib import Path
import re
import json
import torch
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from llama_cpp import Llama
from gptqmodel import GPTQModel  # Ensure GPTQModel is installed or available
from awq import AutoAWQForCausalLM

logger = logging.getLogger(__name__)


ModelLoaderType = Literal["gptq", "gguf", "awq"]


def load_model_config(
    model_name: str, yaml_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load model configuration from a YAML file.

    Args:
        model_name (str): model name in models.yaml file.
        yaml_path (Optional[Path]): Path to the YAML config.
            Defaults to models.yaml in project root.

    Returns:
        Dict[str, Any]: Parsed config dictionary.

    Raises:
        FileNotFoundError: If the file cannot be found or parsed.
        ValueError: If required fields are missing.
    """
    config_path = yaml_path or (Path(__file__).parent.parent / "models.yaml")
    try:
        with open(config_path, "r") as f:
            all_configs = yaml.safe_load(f)
        if model_name not in all_configs:
            raise ValueError(f"Model name '{model_name}' not found in {config_path}")

        logger.info(f"model config ({config_path}) loaded.")
        return all_configs[model_name]
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model config from {config_path}: {e}")


class LLMManager:
    """
    Loads and manages local LLMs (GPTQ, GGUF, or AWQ) for inference, with prompt formatting
    and generation support.
    """

    def __init__(self, use_openai: bool = False, model_name: Optional[str] = None):
        """
        Initialize the LLMManager.

        Args:
            use_openai (bool): Whether to use OpenAI backend instead of local models.
        """
        self.use_openai = use_openai
        self.loader: Optional[ModelLoaderType] = None
        self.tokenizer: Optional[Union[PreTrainedTokenizerBase, Llama]] = None
        self.model: Optional[Any] = None

        if self.use_openai:
            print("[LLMManager] ⚠️ Using OpenAI backend, skipping local model loading.")
            return

        model_name = model_name or "default"
        config = load_model_config(model_name)
        self.loader = config.get("loader", "auto").lower()  # type: ignore
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._load_model(config)

    def _load_gptq_model(self, model_path: Path, dtype: torch.dtype) -> None:
        """
        Load a GPTQ quantized model.

        Args:
            model_path (Path): Path to the GPTQ model directory.
            dtype (torch.dtype): Data type to use for inference.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), use_fast=True, local_files_only=True
        )
        self.model = GPTQModel.from_quantized(
            str(model_path),
            device_map="auto",
            torch_dtype=dtype,
            local_files_only=True,
        )
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerBase
        )  # Ensure it's not Llama tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_gguf_model(self, model_path: Path) -> None:
        """
        Load a GGUF llama.cpp model.

        Args:
            model_path (Path): Path to the GGUF model.
        """
        n_gpu_layers = -1 if self.device == "cuda" else 0
        self.model = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            chat_format="zephyr",
            verbose=True,
        )
        self.tokenizer = self.model

    def _load_awq_model(self, model_path: Path, dtype: torch.dtype) -> None:
        """
        Load an AWQ quantized model from the given path.

        Args:
            model_path (Path): Path to the quantized model directory.
            dtype (torch.dtype): Torch data type to use.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        self.model = AutoAWQForCausalLM.from_quantized(
            str(model_path), device=self.device, torch_dtype=dtype
        )

        if (
            isinstance(self.tokenizer, PreTrainedTokenizerBase)
            and self.tokenizer.pad_token_id is None
        ):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self, config: Dict[str, Any]) -> None:
        """
        Load the model and tokenizer based on the config.

        Args:
            config (Dict[str, Any]): Model configuration dictionary.

        Raises:
            ValueError: If the loader type is unsupported.
        """
        model_path = config["model_path"]
        torch_dtype = config.get("torch_dtype", "float16")
        use_safetensors = config.get("use_safetensors", False)
        dtype = getattr(torch, torch_dtype, torch.float16)

        model_path = Path(model_path)
        if not model_path.is_absolute():
            root_dir = Path(__file__).parent.parent
            model_path = root_dir / model_path
        model_path = model_path.resolve()

        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            f"[LLMManager] ✅ Using model: {model_path} ({self.loader}) on {self.device}"
        )

        if self.loader == "gptq":
            self._load_gptq_model(model_path, dtype)
        elif self.loader == "gguf":
            self._load_gguf_model(model_path)
        elif self.loader == "awq":
            self._load_awq_model(model_path, dtype)
        else:
            raise ValueError(f"Unsupported loader: {self.loader}")

    def _generate_gptq(
        self, prompt: str, system_prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerBase
        )  # Ensure we're using a Hugging Face tokenizer, not llama.cpp

        full_prompt = f"[SYSTEM] {system_prompt}\n[USER] {prompt}\n[ASSISTANT]"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if decoded.startswith(full_prompt):
            return decoded[len(full_prompt) :].strip()
        return decoded

    def _generate_gguf(self, messages: list, max_new_tokens: int) -> str:
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."

        output = self.model.create_chat_completion(
            messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.85,
            stop=["</s>"],
        )
        content = output["choices"][0]["message"]["content"].strip()
        match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        return match.group(0) if match else "[]"

    def _generate_awq(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerBase
        )  # Ensure we're using a Hugging Face tokenizer, not llama.cpp

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature
                > 0.0,  # Only sample stochastically if temp > 0. Else, use greedy decoding
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response using the loaded model.

        Args:
            prompt (str): User prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            system_prompt (Optional[str]): Optional system prompt for instruction tuning.

        Returns:
            str: Generated response.
        """
        if self.use_openai:
            raise RuntimeError("OpenAI mode is active — local generation disabled.")

        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model or tokenizer not initialized."

        system_prompt = system_prompt or (
            'Return only a valid JSON array of tool calls, like [{"tool": "tool_name", "params": {}}]. '
            "No explanations or extra text."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        logger.info(f"[LLMManager] Messages:\n{json.dumps(messages, indent=2)}\n")

        if self.loader == "gptq":
            return self._generate_gptq(
                prompt, system_prompt, max_new_tokens, temperature
            )
        elif self.loader == "gguf":
            return self._generate_gguf(messages, max_new_tokens)
        elif self.loader == "awq":
            return self._generate_awq(prompt, max_new_tokens, temperature)
        else:
            raise ValueError(f"Unsupported loader type: {self.loader}")
