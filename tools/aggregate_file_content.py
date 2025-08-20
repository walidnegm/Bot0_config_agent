"""
tools/aggregate_file_content.py
Combines multiple file contents from prior steps for summarization.
"""

from typing import List
import logging
from tools.tool_models import AggregateFileContentOutput, StepStatus

logger = logging.getLogger(__name__)

def aggregate_file_content(steps: List[Any]) -> AggregateFileContentOutput:
    """
    Aggregate contents from previous step results.
    Assumes each step is a dict (e.g., from read_files: {path: content}); concatenates the contents.
    Handles None or non-dict as empty string.
    """
    aggregated = ''
    for step in steps or []:
        if step is None:
            continue
        if isinstance(step, dict):
            aggregated += '\n\n'.join(step.values()) + '\n'
        else:
            aggregated += str(step) + '\n'
    return AggregateFileContentOutput(status=StepStatus.SUCCESS, message="Aggregated file contents.", result=aggregated.strip())
