"""bot0_config_agent/tools/tool_scripts/find_files_by_keywords.py"""

import os
from bot0_config_agent.agent_models.step_status import StepStatus


def find_files_by_keywords(**kwargs):
    """
    Finds files in a directory that contain specific keywords in their names.

    This function searches for files within a specified root directory (or the
    current working directory by default) and its subdirectories. It matches
    files if their names (case-insensitive) contain any of the provided keywords.

    Args:
        **kwargs: Arbitrary keyword arguments.
            - keywords (Union[str, list[str]]): A single keyword or a list of
            keywords to search for in file names. This parameter is required.
            - dir (str, optional): The starting directory for the search.
            Defaults to the current working directory.

    Returns:
        dict: A dictionary with the status and results of the operation.
              On success, it includes a list of relative paths to all matching files.
              On failure, it provides an error status and message.
    """
    keywords = kwargs.get("keywords")
    dir = kwargs.get("dir", os.getcwd())

    if not keywords:
        return {
            "status": StepStatus.ERROR,
            "message": "Missing required parameter: keywords",
        }

    if isinstance(keywords, str):
        keywords = [keywords]

    keywords = [k.lower() for k in keywords]
    matches = []

    for dirpath, _, files in os.walk(dir):
        for f in files:
            if any(k in f.lower() for k in keywords):
                matches.append(os.path.relpath(os.path.join(dirpath, f), start=dir))

    return {
        "status": StepStatus.SUCCESS,
        "message": f"Found {len(matches)} file(s) matching keywords: {', '.join(keywords)}",
        "matches": matches,
    }
