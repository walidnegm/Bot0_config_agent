import os
from agent_models.step_status import StepStatus


def find_file_by_keyword(**kwargs):
    keywords = kwargs.get("keywords")
    root = kwargs.get("root", os.getcwd())

    if not keywords:
        return {
            "status": StepStatus.ERROR,
            "message": "Missing required parameter: keywords",
        }

    if isinstance(keywords, str):
        keywords = [keywords]

    keywords = [k.lower() for k in keywords]
    matches = []

    for dirpath, _, files in os.walk(root):
        for f in files:
            if any(k in f.lower() for k in keywords):
                matches.append(os.path.relpath(os.path.join(dirpath, f), start=root))

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Found {len(matches)} file(s) matching keywords: {', '.join(keywords)}",
        "matches": matches,
    }
