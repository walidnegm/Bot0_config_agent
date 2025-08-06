# utils/gpu_monitor.py
import logging
import pynvml
import torch
import logging_config  # Make sure this sets up "resource" logger

resource_logger = logging.getLogger(__name__)


def log_gpu_usage(step_name: str = ""):
    """
    Log current GPU VRAM usage (via NVML and PyTorch) with an optional step label.

    Useful for profiling and debugging memory use when loading large models or running inference.
    Outputs include both NVIDIA driver-reported usage (NVML) and PyTorch's own allocator stats.

    Args:
        step_name (str): Label for this measurement (e.g., "before_load_model", 
            "after_inference", "OOM").

    Example output:
        [VRAM][before_load_model] NVML: 612.12/4096.00 MB | PyTorch: alloc=4.20 MB, reserved=4.20 MB, \
max_alloc=4.20 MB, max_reserved=4.20 MB

    How to use:
        # Import at the top of your file
        from utils.gpu_monitor import log_gpu_usage

        # Log before and after major steps (model load, inference, cache clearing)
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
        - Logs are sent to the dedicated "resource" logger.
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
        resource_logger = logging.getLogger("resource")
        resource_logger.info(
            f"[VRAM][{step_name}] NVML: {used_mb:.2f}/{total_mb:.2f} MB | "
            f"PyTorch: alloc={torch_alloc:.2f} MB, reserved={torch_reserved:.2f} MB, "
            f"max_alloc={torch_max_alloc:.2f} MB, max_reserved={torch_max_reserved:.2f} MB"
        )
    except Exception as e:
        resource_logger = logging.getLogger("resource")
        resource_logger.error(f"Failed to log GPU usage: {e}")
