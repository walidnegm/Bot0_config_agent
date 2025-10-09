"""
MCP tool: checks CUDA availability, version, and GPU information using PyTorch and nvidia-smi.
"""

import subprocess
import torch


def main() -> dict:
    """Return CUDA and GPU details using PyTorch and NVIDIA-SMI."""
    result = {
        "status": "success",
        "message": "",
        "result": {}
    }

    try:
        # --- PyTorch CUDA info ---
        is_available = torch.cuda.is_available()
        result["result"]["cuda_available"] = is_available
        result["result"]["torch_cuda_version"] = torch.version.cuda

        if is_available:
            result["result"]["cuda_device"] = torch.cuda.get_device_name(0)

        # --- NVIDIA-SMI info ---
        try:
            smi_output = subprocess.check_output(
                ["nvidia-smi"],
                stderr=subprocess.STDOUT,
                text=True
            )
            lines = smi_output.strip().splitlines()
            result["result"]["nvidia_smi_summary"] = lines[0] if lines else "n/a"
            result["result"]["nvidia_smi"] = "\n".join(lines[:20]) + "\n... (truncated)"
        except FileNotFoundError:
            result["result"]["nvidia_smi"] = "nvidia-smi not found"
        except subprocess.CalledProcessError as e:
            result["result"]["nvidia_smi"] = f"nvidia-smi error: {e.output.strip()}"

        result["message"] = "CUDA check completed successfully."

    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result


# ✅ MCP metadata definition
def get_tool_definition():
    return {
        "name": "check_cuda",
        "description": "Checks CUDA availability, version, and GPU details using PyTorch and nvidia-smi.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


# ✅ Entry point exposed to MCP discovery
def run(params):
    return main()

