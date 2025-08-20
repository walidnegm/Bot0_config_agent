from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union  # Added Any to fix the error
from pydantic import BaseModel, Field, field_validator, model_validator
from agent_models.step_state import StepState
from agent_models.step_status import StepStatus


# -----------------------------------------------------------------------------
# Shared wrappers
# -----------------------------------------------------------------------------
class ToolResult(BaseModel):
    """
    Represents the result of a single tool call after execution.
    """
    step_id: str
    tool: str
    params: Dict[str, Any]
    message: Optional[str] = ""
    status: Optional[StepStatus] = None
    result: Optional[Any] = None
    state: StepState = StepState.PENDING


class ToolResults(BaseModel):
    """
    Container for all results returned from executing a multi-step tool plan.
    """
    results: List[ToolResult]


class ToolOutput(BaseModel):
    """
    Standard envelope shape for tool outputs.
    """
    status: StepStatus
    message: Optional[str] = None
    result: Any  # Another Pydantic model or primitive


# -----------------------------------------------------------------------------
# aggregate_file_content
# -----------------------------------------------------------------------------
class AggregateFileContentInput(BaseModel):
    steps: List[str]


class AggregateFileContentOutput(ToolOutput):
    result: Optional[str] = None  # aggregated text blob


# -----------------------------------------------------------------------------
# check_cuda
# -----------------------------------------------------------------------------
class CheckCudaInput(BaseModel):
    pass


class CheckCudaResult(BaseModel):
    cuda_available: bool


class CheckCudaOutput(ToolOutput):
    result: Optional[CheckCudaResult] = None


# -----------------------------------------------------------------------------
# echo_message
# -----------------------------------------------------------------------------
class EchoMessageInput(BaseModel):
    message: str


class EchoMessageOutput(ToolOutput):
    result: Optional[str] = None  # echoed message


# -----------------------------------------------------------------------------
# find_dir_size
# -----------------------------------------------------------------------------
class FindDirSizeInput(BaseModel):
    root: Optional[str] = None


class FindDirSizeResult(BaseModel):
    file_count: int
    total_size_bytes: int
    total_size_mb: float


class FindDirSizeOutput(ToolOutput):
    result: Optional[FindDirSizeResult] = None


# -----------------------------------------------------------------------------
# find_dir_structure
# -----------------------------------------------------------------------------
NodeType = Literal["directory", "file"]


class DirectoryNode(BaseModel):
    """
    One node in the directory tree.
    - For files: `type="file"`, no children.
    - For directories: `type="directory"`, children is a (possibly empty) list.
    """
    name: str = Field(..., description="Base name of file or directory")
    type: NodeType = Field(..., description="Node type: 'file' or 'directory'")
    path: str = Field(..., description="Relative path from the root")
    children: List[DirectoryNode] = Field(
        default_factory=list, description="Child nodes (for directories only)"
    )


class FindDirStructureResult(BaseModel):
    tree: DirectoryNode


class FindDirStructureOutput(ToolOutput):
    result: Optional[FindDirStructureResult] = None


# -----------------------------------------------------------------------------
# find_file_by_keyword
# -----------------------------------------------------------------------------
class FindFileByKeywordInput(BaseModel):
    keyword: str
    root: Optional[str] = None


class FindFileByKeywordOutput(ToolOutput):
    result: Optional[List[str]] = None  # list of matching file paths


# -----------------------------------------------------------------------------
# list_project_files
# -----------------------------------------------------------------------------
class ListProjectFilesInput(BaseModel):
    root: Optional[str] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class ListProjectFilesOutput(ToolOutput):
    result: Optional[List[str]] = None  # list of file paths


# -----------------------------------------------------------------------------
# llm_response_async
# -----------------------------------------------------------------------------
class LLMResponseInput(BaseModel):
    prompt: str


class LLMResponseResult(BaseModel):
    text: str


class LLMResponseOutput(ToolOutput):
    result: Optional[LLMResponseResult] = None


# -----------------------------------------------------------------------------
# locate_file
# -----------------------------------------------------------------------------
class LocateFileInput(BaseModel):
    filename: str
    root: Optional[str] = None


class LocateFileOutput(ToolOutput):
    result: Optional[str] = None  # file path or None

# -----------------------------------------------------------------------------
# locate_file
# -----------------------------------------------------------------------------
class LocateFileResult(BaseModel):
    path: str


class LocateFileOutput(ToolOutput):
    result: Optional[LocateFileResult] = None
# -----------------------------------------------------------------------------
# make_virtualenv
# -----------------------------------------------------------------------------
class MakeVirtualenvInput(BaseModel):
    path: str


class MakeVirtualenvOutput(ToolOutput):
    result: Optional[str] = None  # path to created virtualenv


# -----------------------------------------------------------------------------
# read_files
# -----------------------------------------------------------------------------
class ReadFilesInput(BaseModel):
    path: List[str]


class ReadFilesResult(BaseModel):
    contents: Dict[str, str]  # path -> content


class ReadFilesOutput(ToolOutput):
    result: Optional[ReadFilesResult] = None


# -----------------------------------------------------------------------------
# retrieval_tool
# -----------------------------------------------------------------------------
class RetrievalToolInput(BaseModel):
    query: str


class RetrievalToolOutput(ToolOutput):
    result: Optional[List[str]] = None  # e.g., matched keys/paths


# -----------------------------------------------------------------------------
# seed_parser
# -----------------------------------------------------------------------------
class SeedParserInput(BaseModel):
    file: str


class SeedParserResult(BaseModel):
    parsed_data: Dict[str, Any]


class SeedParserOutput(ToolOutput):
    result: Optional[SeedParserResult] = None


# -----------------------------------------------------------------------------
# set_scope
# -----------------------------------------------------------------------------
class SetScopeInput(BaseModel):
    """
    Input model for the `set_scope` tool.
    - root: project root directory (absolute or relative). Optional.
    - branches: one or more subpaths under root. Accepts str or list[str].
    """
    root: Optional[str] = None
    branches: List[str] = Field(default_factory=list)

    @field_validator("branches", mode="before")
    @classmethod
    def _coerce_branches(cls, v: Union[None, str, List[str]]) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        raise TypeError("`branches` must be a string, list of strings, or None")


class SetScopePayload(BaseModel):
    """
    Payload returned under `result` by the `set_scope` tool.
    """
    root: Optional[str] = None
    branches: List[str] = Field(default_factory=list)


class SetScopeOutput(ToolOutput):
    """
    Standard tool output envelope for `set_scope`.
    - status/message inherited from ToolOutput
    - result contains { root, branches }
    """
    result: Optional[SetScopePayload] = None


# Registry expects "...Result" naming; keep an alias/subclass for compatibility.
class SetScopeResult(SetScopeOutput):
    pass


# -----------------------------------------------------------------------------
# summarize_config
# -----------------------------------------------------------------------------
class SummarizeConfigInput(BaseModel):
    pass


class SummarizeConfigOutput(ToolOutput):
    result: Optional[str] = None  # summary text


# -----------------------------------------------------------------------------
# summarize_files
# -----------------------------------------------------------------------------
class SummarizeFilesInput(BaseModel):
    files: List[str]


class SummarizeFilesOutput(ToolOutput):
    result: Optional[str] = None  # summary text


# -----------------------------------------------------------------------------
# ToolSpec (optional metadata for registry)
# -----------------------------------------------------------------------------
class ToolSpec(BaseModel):
    description: str
    import_path: str
    parameters: Dict[str, Any]  # keep JSON Schema here if you want
    input_model: Optional[str] = None  # e.g. "ReadFilesInput"
    output_model: Optional[str] = None  # e.g. "ReadFilesResult"
