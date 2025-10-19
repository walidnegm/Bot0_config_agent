"""
tools/locate_file.py
Locates a file by name in a directory.
"""

import os
from pathlib import Path
import logging
from tools.tool_models import LocateFileOutput, LocateFileResult, StepStatus  # Added LocateFileResult

logger = logging.getLogger(__name__)

def locate_file(filename: str, root: str = '.') -> LocateFileOutput:
    """
    Search for a file by exact name in the given root directory and subdirectories (excluding __pycache__).
    Returns the first matching path in structured result; message notes if multiple found.
    """
    matches = []
    root_path = Path(root).resolve()
    for dirpath, dirnames, filenames in os.walk(root_path):
        if '__pycache__' in dirpath:
            continue  # Skip __pycache__
        if filename in filenames:
            matches.append(os.path.join(dirpath, filename))
    if not matches:
        return LocateFileOutput(status=StepStatus.ERROR, message=f"No match found for '{filename}'.", result=None)
    message = f"Found {len(matches)} match(es) for '{filename}'."
    if len(matches) > 1:
        message += " Returning the first match."
    return LocateFileOutput(status=StepStatus.SUCCESS, message=message, result=LocateFileResult(path=matches[0]))
