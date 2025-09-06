# utils/gpu_monitor.py
import logging
import pynvml
import torch

# import logging_config  # Make sure this sets up "resource" logger

logger = logging.getLogger(__name__)


def log_gpu_usage(step_name: str = "") -> float:
    """
    Log current GPU VRAM usage (via NVML and PyTorch) with an optional step label.

    Useful for profiling and debugging memory use when loading large models or running inference.
    Outputs include both NVIDIA driver-reported usage (NVML) and PyTorch's own allocator stats.

    Args:
        step_name (str): Label for this measurement (e.g., "before_load_model", 
            "after_inference", "OOM").

    Returns:
        used_mb (float): Current VRAM memory usage in MB (useful for calculations).

    Example output:
        [VRAM][before_load_model] NVML: 612.12/4096.00 MB | PyTorch: alloc=4.20 MB, reserved=4.20 MB, \
max_alloc=4.20 MB, max_reserved=4.20 MB

    How to use:
        # Import at the top of your file
        from utils.system.gpu_monitor import log_gpu_usage

        log_gpu_usage("before_load_model")
        llm = get_llm_manager(self.local_model_name)
        log_gpu_usage("after_load_model")

        log_gpu_usage("before_inference")
        output = llm.generate(...)
        log_gpu_usage("after_inference")

        # On CUDA OOM
        try:
            ...
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                log_gpu_usage("OOM")
            raise

    Notes:
        - Requires `pynvml` and `torch` to be installed.
        - Logs are saved to the main log file.
        - If called on a non-GPU machine or without drivers, will log an error (but not crash).
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = int(info.used) / (1024**2)
        total_mb = int(info.total) / (1024**2)
        torch_alloc = torch.cuda.memory_allocated() / (1024**2)
        torch_reserved = torch.cuda.memory_reserved() / (1024**2)
        torch_max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        torch_max_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        logger.info(
            f"[VRAM][{step_name}] NVML: {used_mb:.2f}/{total_mb:.2f} MB | "
            f"PyTorch: alloc={torch_alloc:.2f} MB, reserved={torch_reserved:.2f} MB, "
            f"max_alloc={torch_max_alloc:.2f} MB, max_reserved={torch_max_reserved:.2f} MB"
        )
        return used_mb
    except Exception as e:
        logger.error(f"Failed to log GPU usage: {e}")
        return -1.0  # Indicates logging failed or unavailable


def log_peak_vram_usage(label: str = ""):
    """
    Logs the peak VRAM allocated by PyTorch in MB since the last reset.

    Args:
        label (str): Custom label for this measurement (e.g., "after_load", "after_inference").

    Return:
        peak_mb (float): peak VRAM memory usage (in case needed for other calcuations).

    How to use:
        # Before the code block you want to profile:
        torch.cuda.reset_peak_memory_stats()

        # ...your code (e.g., model load or inference)...

        log_peak_vram_usage("after_load_model")

    Example log:
        [PEAK_VRAM][after_load_model] PyTorch peak allocated: 2240.55 MB

    Notes:
        - Only shows *your process’s* max PyTorch VRAM usage.
        - Call `torch.cuda.reset_peak_memory_stats()` *before* the code you want to measure.
    """
    try:
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(...)
        return peak_mb
    except Exception as e:
        logger.error(...)
        return 0.0  # or -1.0 to indicate an error


def log_embedding_footprint(model, label: str = "embeddings"):
    """
    Report total VRAM footprint attributable to embedding modules by summing param sizes.
    Works for standard and quantized models (falls back to any *Embedding* class name).
    """
    try:
        import torch.nn as nn
        import torch

        total_bytes = 0
        rows = []

        for name, mod in model.named_modules():
            cls = mod.__class__.__name__
            is_emb = isinstance(mod, nn.Embedding) or ("Embedding" in cls)
            if not is_emb:
                continue
            p = next((p for p in mod.parameters(recurse=False)), None)
            if p is None:
                continue
            b = p.numel() * p.element_size()
            total_bytes += b
            rows.append(
                f"{name} ({cls}) — {p.numel():,} elems, {b/1024**2:.1f} MB, dtype={p.dtype}, device={p.device}"
            )

        mb = total_bytes / (1024**2)
        logger.info("[EMB][%s] %d module(s), total ~= %.1f MB", label, len(rows), mb)
        for r in rows:
            logger.info("[EMB] %s", r)
        return mb
    except Exception as e:
        logger.error(f"[EMB] Failed to log embedding footprint: {e}")
        return 0.0
