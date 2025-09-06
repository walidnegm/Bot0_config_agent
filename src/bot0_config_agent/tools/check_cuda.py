# tools/check_cuda.py
# -------------------
# Probe CUDA availability via PyTorch and (optionally) nvidia-smi.

import subprocess
import torch
from bot0_config_agent.agent_models.step_status import StepStatus


def check_cuda():
    """
    Check CUDA/GPU availability and basic environment information.

    Returns:
        dict: Standard tool envelope in the exact field order:
            {
                "status": StepStatus,         # SUCCESS or ERROR
                "message": str,               # human-readable summary
                "result": {                   # payload (None on fatal error)
                    "cuda_available": bool,
                    "torch_version": str | None,
                    "torch_cuda_version": str | None,   # build-time CUDA (may be None)
                    "cuda_device": str | None,          # first device name if available
                    "device_count": int,                # number of CUDA devices
                    "nvidia_smi_summary": str | None,   # first line of nvidia-smi
                    "nvidia_smi": str | None            # first ~20 lines or message
                }
            }
    Notes:
        - torch.version.cuda can be None if PyTorch was built without CUDA.
        - nvidia-smi may not exist on the system; we handle that gracefully.
    """
    # Keep literal key order: status, message, result
    status = StepStatus.SUCCESS
    message = ""
    payload = {
        "cuda_available": False,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_device": None,
        "device_count": 0,
        "nvidia_smi_summary": None,
        "nvidia_smi": None,
    }

    try:
        # PyTorch view of CUDA
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["device_count"] = int(torch.cuda.device_count())

        if payload["cuda_available"] and payload["device_count"] > 0:
            # Get the name of device 0 safely
            try:
                payload["cuda_device"] = torch.cuda.get_device_name(0)
            except Exception:
                payload["cuda_device"] = "Unknown CUDA device 0"

        # Try to run nvidia-smi (optional)
        try:
            smi_output = subprocess.check_output(
                ["nvidia-smi"], stderr=subprocess.STDOUT, text=True
            )
            lines = [ln.rstrip() for ln in smi_output.splitlines()]
            if lines:
                payload["nvidia_smi_summary"] = lines[0]
                # show first 20 lines max
                head = lines[:20]
                if len(lines) > 20:
                    head.append("... (truncated)")
                payload["nvidia_smi"] = "\n".join(head)
        except FileNotFoundError:
            payload["nvidia_smi"] = "nvidia-smi not found"
        except subprocess.CalledProcessError as e:
            payload["nvidia_smi"] = f"nvidia-smi error: {e.output.strip()}"

        # Build a concise message
        if payload["cuda_available"]:
            dev = payload["cuda_device"] or "Unknown device"
            message = (
                f"CUDA available ({payload['device_count']} device(s)); first: {dev}."
            )
        else:
            message = "CUDA not available in PyTorch."

    except Exception as e:
        status = StepStatus.ERROR
        message = f"CUDA check failed: {e}"
        payload = None  # explicit None on fatal error

    # Return in canonical order
    return {
        "status": status,
        "message": message,
        "result": payload,
    }
