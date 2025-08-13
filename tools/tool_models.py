from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from agent_models.step_state import StepState
from agent_models.step_status import StepStatus


class ToolResult(BaseModel):  # Optional for now; use later
    """
    Represents the result of a single tool call after execution.

    Attributes:
        step_id (str): step_0, step_1, etc.
        tool (str): Tool name executed.
        params (Dict[str, Any]): Parameters passed to the tool.
        message (Optional[str]): Optional message (error details or output summary).
        status (Optional[StepStatus]): Success/error after execution, or None before run.
        result (Optional[Any]): Result returned from the tool (list, dict, str, etc).
        state (StepState): Execution lifecycle state (pending, in_progress, completed,
            failed).

    Example:
        {
            "step_id": "step_0"
            "tool": "list_project_files",
            "params": {"root": "."},
            "status": "success",
            "message": "",
            "result": ["file1.py", "file2.md"]
        }
    """

    step_id: str
    tool: str
    params: Dict[str, Any]
    message: Optional[str] = ""
    status: Optional[StepStatus] = None  # None before execution
    result: Optional[Any] = None
    state: StepState = StepState.PENDING


class ToolResults(BaseModel):
    """
    Container for all results returned from executing a multi-step tool plan.

    Attributes:
        results (List[ToolResult]): List of results from each executed tool call.

    Example:
        {
            "results": [
                {... ToolResult ...},
                {... ToolResult ...}
            ]
        }
    """

    results: List[ToolResult]


class ToolOutput(BaseModel):  # Wrapper for output
    status: StepStatus
    message: Optional[str] = None
    result: Any  # Anoither Pydantic model or primitive


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

    name: str = Field(..., description="Base name of the file or directory.")
    type: NodeType = Field(..., description="'directory' or 'file'.")
    children: Optional[List["DirectoryNode"]] = Field(
        default=None, description="Present only when type == 'directory'."
    )

    @model_validator(mode="after")
    def _enforce_children_rules(self) -> "DirectoryNode":
        if self.type == "file" and self.children is not None:
            # Keep API strict so your downstream logic is predictable
            raise ValueError("Files must not include 'children'.")
        if self.type == "directory" and self.children is None:
            # The implementation always sets children=[] for directories,
            # but allow either empty list or we auto-normalize:
            object.__setattr__(self, "children", [])
        return self


DirectoryNode.model_rebuild()  # for forward refs


class FindDirStructureResult(BaseModel):
    """
    Wrapper returned by tools.find_dir_structure.find_dir_structure().
    """

    status: StepStatus
    message: str = Field("", description="Summary or error message.")
    result: Optional[DirectoryNode] = Field(
        None, description="Directory tree when status=='success'; None on error."
    )


# -----------------------------------------------------------------------------
# find_file_by_keyword
# -----------------------------------------------------------------------------
class FindFileByKeywordInput(BaseModel):
    keywords: List[str]
    root: Optional[str] = None


class FindFileByKeywordResult(BaseModel):
    files: List[str]


class FindFileByKeywordOutput(ToolOutput):
    result: Optional[FindFileByKeywordResult] = None


# -----------------------------------------------------------------------------
# list_project_files
# -----------------------------------------------------------------------------
class ListProjectFilesResult(BaseModel):
    files: List[str]


class ListProjectFilesOutput(ToolOutput):
    result: ListProjectFilesResult


# -----------------------------------------------------------------------------
# llm_response_async
# -----------------------------------------------------------------------------
class LlmResponseAsyncInput(BaseModel):
    prompt: str


class LlmResponseAsyncOutput(ToolOutput):
    result: Optional[str] = None  # generated content


# -----------------------------------------------------------------------------
# locate_file
# -----------------------------------------------------------------------------
class LocateFileInput(BaseModel):
    filename: str


class LocateFileResult(BaseModel):
    path: Optional[str] = None


class LocateFileOutput(ToolOutput):
    result: Optional[LocateFileResult] = None


# -----------------------------------------------------------------------------
# make_virtualenv
# -----------------------------------------------------------------------------
class MakeVirtualenvInput(BaseModel):
    path: str


class MakeVirtualenvOutput(ToolOutput):
    result: Optional[dict] = None  # keep generic (e.g., {"created": true})


# -----------------------------------------------------------------------------
# read_files  (supports single str or list[str] in input; output is list items)
# -----------------------------------------------------------------------------
class ReadFilesInput(BaseModel):
    path: Union[str, List[str]]

    @field_validator("path", mode="before")
    @classmethod
    def wrap_single_path(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class ReadFileItem(BaseModel):
    file: str
    content: str


class ReadFilesOutput(ToolOutput):
    result: Optional[List[ReadFileItem]] = None


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
            # ensure all are strings
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
class ToolSpec(BaseModel):
    description: str
    import_path: str
    parameters: Dict[str, Any]  # keep JSON Schema here if you want
    input_model: Optional[str] = None  # "agent_models.tools.ReadFilesInput"
    output_model: Optional[str] = None  # "agent_models.tools.ReadFilesOutput"
