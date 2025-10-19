"""
tools/aggregate_file_content.py
Combines multiple file contents from prior steps for summarization.
"""

from typing import List, Any
import logging
from tools.tool_models import AggregateFileContentOutput
from agent_models.step_status import StepStatus

logger = logging.getLogger(__name__)

def _extract_text_from_step(step: Any) -> str:
    """
    Normalize various step output shapes to a single text blob.
    Supports:
      - {"contents": {path: content, ...}}
      - {"files": [{"path": ..., "content": ...}, ...]}
      - arbitrary dicts (joins values)
      - primitives -> str()
    """
    if step is None:
        return ""
    if isinstance(step, dict):
        # read_files newer shape
        if "files" in step and isinstance(step["files"], list):
            parts = []
            for item in step["files"]:
                try:
                    if isinstance(item, dict) and "content" in item:
                        parts.append(str(item["content"]))
                except Exception:
                    continue
            return "\n".join(parts).strip()
        # read_files older shape
        if "contents" in step and isinstance(step["contents"], dict):
            return "\n".join(str(v) for v in step["contents"].values()).strip()
        # generic dict: join values
        try:
            return "\n".join(str(v) for v in step.values()).strip()
        except Exception:
            return str(step)
    # lists -> join recursively
    if isinstance(step, list):
        return "\n".join(_extract_text_from_step(s) for s in step).strip()
    # primitives
    return str(step)

def aggregate_file_content(steps: List[Any]) -> AggregateFileContentOutput:
    """
    Aggregate contents from previous step results.
    Assumes each 'step' entry is either a prior step's result object or a piece of text.
    """
    text_blobs = []
    for step in steps or []:
        text_blobs.append(_extract_text_from_step(step))
    aggregated = "\n\n".join([t for t in text_blobs if t]).strip()

    return AggregateFileContentOutput(
        status=StepStatus.SUCCESS,
        message="Aggregated file contents.",
        result=aggregated
    )

