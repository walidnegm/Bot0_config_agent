from agent_models.step_status import StepStatus


def make_virtualenv(**kwargs):
    import venv
    import os

    path = kwargs.get("path")
    if not path:
        return {
            "status": StepStatus.ERROR,
            "message": "Missing required parameter: path",
        }

    try:
        os.makedirs(path, exist_ok=True)
        venv.create(path, with_pip=True)
        return {
            "status": StepStatus.SUCCESS,
            "message": f"Virtualenv created at {path}",
        }
    except Exception as e:
        return {"status": StepStatus.ERROR, "message": str(e)}
