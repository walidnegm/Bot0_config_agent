import torch
import subprocess

def check_cuda():
    result = {
        "status": "ok",
        "result": {},
        "message": ""
    }

    try:
        # Check CUDA availability
        is_available = torch.cuda.is_available()
        result["result"]["cuda_available"] = is_available
        result["result"]["torch_cuda_version"] = torch.version.cuda

        # Optionally get device name if available
        if is_available:
            result["result"]["cuda_device"] = torch.cuda.get_device_name(0)

        # Try to run `nvidia-smi`
        try:
            smi_output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
            result["result"]["nvidia_smi_summary"] = smi_output.splitlines()[0]
            result["result"]["nvidia_smi"] = "\n".join(smi_output.strip().splitlines()[:20]) + "\n... (truncated)"
        except FileNotFoundError:
            result["result"]["nvidia_smi"] = "nvidia-smi not found"
        except subprocess.CalledProcessError as e:
            result["result"]["nvidia_smi"] = f"nvidia-smi error: {e.output.strip()}"

    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return result

