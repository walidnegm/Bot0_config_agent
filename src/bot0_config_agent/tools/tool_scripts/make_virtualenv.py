# bot0_config_agent/tools/tool_scripts/make_virtualenv.py
# ------------------------
# Create a Python virtual environment at a given directory.


import os
import venv
from bot0_config_agent.agent_models.step_status import StepStatus


def make_virtualenv(**kwargs):
    """
    Create a Python virtual environment at the given directory.

    Args (kwargs):
        dir (str): Target directory where the virtual environment will be created.

    Returns:
        dict: Standard tool envelope in the order (status, message, result)
              result = {"created": bool, "dir": str}
    """
    dir_path = kwargs.get("dir")
    if not dir_path or not isinstance(dir_path, str):
        return {
            "status": StepStatus.ERROR,
            "message": "Missing required parameter: 'dir' (str).",
            "result": {"created": False, "dir": None},
        }

    # Normalize path
    dir_path = os.path.abspath(os.path.expanduser(dir_path))

    try:
        os.makedirs(dir_path, exist_ok=True)

        # If it already looks like a venv, report success without recreating
        pyvenv_cfg = os.path.join(dir_path, "pyvenv.cfg")
        if os.path.isfile(pyvenv_cfg):
            return {
                "status": StepStatus.SUCCESS,
                "message": f"Virtualenv already exists at {dir_path}.",
                "result": {"created": False, "dir": dir_path},
            }

        # Create the environment with pip
        venv.create(dir_path, with_pip=True)

        return {
            "status": StepStatus.SUCCESS,
            "message": f"Virtualenv created at {dir_path}.",
            "result": {"created": True, "dir": dir_path},
        }

    except Exception as e:
        return {
            "status": StepStatus.ERROR,
            "message": f"Failed to create virtualenv: {e}",
            "result": {"created": False, "dir": dir_path},
        }
