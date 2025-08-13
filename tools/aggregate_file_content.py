# tools/aggregate_file_content.py
# --------------------------------
# Combine multiple file contents from prior steps for summarization

from agent_models.step_status import StepStatus


def aggregate_file_content(**kwargs):
    steps = kwargs.get("steps", [])  # Expected to be resolved step values

    if not steps or not isinstance(steps, list):
        return {
            "status": StepStatus.ERROR,
            "message": "Missing or invalid 'steps' parameter. Must be a list of file contents.",
        }

    try:
        joined = "\n\n".join(str(s) for s in steps)
        return {
            "status": StepStatus.SUCCESS,
            "message": "Aggregated file contents.",
            "result": joined,
        }
    except Exception as e:
        return {"status": StepStatus.ERROR, "message": f"Failed to aggregate: {e}"}
