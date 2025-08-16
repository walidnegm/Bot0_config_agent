"""tools/workbench/tool_models.py"""

# Standard and 3rd party imports
from __future__ import annotations
from pathlib import Path
import importlib
import logging
from typing import Any, Dict, Iterable, List, Literal, Optional, Union
from enum import Enum
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    field_serializer,
    ConfigDict,
)

# Local imports
from agent_models.step_state import StepState
from agent_models.step_status import StepStatus


logger = logging.getLogger(__name__)


# * Common Tool Models

# =============================================================================
# Common envelopes for tool execution results (per-step)
# =============================================================================


class ToolResult(BaseModel):
    """Per-step execution record used by the executor/CLI."""

    step_id: str
    tool: str
    params: Dict[str, Any]
    message: Optional[str] = ""
    status: Optional[StepStatus] = None
    result: Optional[Any] = None
    state: StepState = StepState.PENDING

    model_config = ConfigDict(use_enum_values=True)


class ToolResults(BaseModel):
    """Container for the full chain execution results."""

    results: List[ToolResult]


class ToolOutput(BaseModel):
    """
    Standard tool output envelope.

    Fields:
      - status: StepStatus (SUCCESS/ERROR)
      - message: optional short human text
      - result: typed payload or primitive
    """

    status: StepStatus
    message: Optional[str] = None
    result: Any

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# ToolSpec (registry schema carriers)
# =============================================================================


class ToolSpec(BaseModel):
    """
    Registry metadata holder (JSON schema-like).
      - description: human-readable tool summary
      - import_path: module.fn to import
      - parameters: JSON-schema dict describing inputs
      - input_model/output_model: optional fully-qualified model names
    """

    description: str
    import_path: str
    parameters: Dict[str, Any]
    input_model: Optional[str] = None
    output_model: Optional[str] = None


# =============================================================================
# ToolTransformation
# =============================================================================


class ToolTransformation(BaseModel):
    """
    Defines a transformation mapping between a source tool's output and a target
    tool's input.

    This model specifies how the output of a source tool is transformed to match
    the input schema of a target tool, enabling tool chaining in workflows.

    Used in 'tool_transformation.json' to centralize transformation logic for
    ToolRegistry.

    Attributes:
        source_tool (str): Name of the source tool (must match a key in
            ToolRegistry.tools).
        target_tool (str): Name of the target tool (must match a key in
            ToolRegistry.tools).
        transform_fn (Optional[str]): Dotted path to a transformation function
                                    (e.g., 'tools.transformation.dir_to_files')
                                    or None for direct matches where no
                                    transformation is needed.
        description (Optional[str]): Description of the transformation's purpose.
    """

    source_tool: str = Field(
        ...,
        description="Name of the source tool whose output is transformed (e.g., 'find_dir_structure').",
    )
    target_tool: str = Field(
        ...,
        description="Name of the target tool whose input schema the output is transformed to match \
(e.g., 'read_files').",
    )
    transform_fn: Optional[str] = Field(
        None,
        description="Dotted path to a Python function for transforming output to input \
(e.g., 'tools.transformation.dir_to_files'). None if no transformation is needed.",
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of the transformation's purpose (e.g., \
'Converts DirectoryNode to List[str] for file reading').",
    )

    @field_validator("transform_fn")
    @classmethod
    def validate_transform_fn(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate that transform_fn is a valid Python function path or None.

        If None, then it is one -> one validation with Pydantic.
        """
        if v is None:
            return v
        dotted = v.strip()

        try:
            module_path, fn_name = dotted.rsplit(".", 1)
        except ValueError:
            raise ValueError(f"Invalid dotted path for transform_fn: {v!r}")

        try:
            module = importlib.import_module(module_path)
            obj = getattr(module, fn_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Invalid transform_fn '{v}': {e}")
            raise ValueError(f"Invalid transform_fn '{v}': {e}")

        if not callable(obj):
            raise ValueError(f"'{v}' is not callable")

        return dotted  # Normalized

    @model_validator(mode="after")
    def validate_tool_names(self):
        """
        Validate that source_tool and target_tool are non-empty strings.

        Note: Validation against ToolRegistry.tools requires registry instance,
        handled in ToolRegistry.
        """
        if not self.source_tool or not isinstance(self.source_tool, str):
            raise ValueError("source_tool must be a non-empty string")
        if not self.target_tool or not isinstance(self.target_tool, str):
            raise ValueError("target_tool must be a non-empty string")
        return self

    # Pydantic configuration for ToolTransformation;
    model_config = ConfigDict(
        extra="forbid",  # Prevent extra fields in the model
        json_schema_extra={
            "example": {
                "source_tool": "find_dir_structure",
                "target_tool": "read_files",
                "transform_fn": "tools.transformation.dir_to_files",
                "description": "Converts DirectoryNode to List[str] for file reading",
            }
        },
    )


# * Specific Tools

# =============================================================================
# aggregate_file_content
# =============================================================================


class AggregateFileContentInput(BaseModel):
    """Input: list of step references to concatenate (e.g., ['<step_0>', '<step_1>'])."""

    steps: List[str]


class AggregateFileContentResult(ToolOutput):
    """Output: aggregated text under result."""

    result: Optional[str] = None


# =============================================================================
# check_cuda
# =============================================================================


class CheckCudaInput(BaseModel):
    """Input: none."""

    pass


class CheckCudaPayload(BaseModel):
    """Payload: CUDA availability info."""

    cuda_available: bool


class CheckCudaResult(ToolOutput):
    """Output: CUDA status under result."""

    result: Optional[CheckCudaPayload] = None


# =============================================================================
# echo_message
# =============================================================================


class EchoMessageInput(BaseModel):
    """Input: the message to echo."""

    message: str


class EchoMessageResult(ToolOutput):
    """Output: echoed message under result."""

    result: Optional[str] = None


# =============================================================================
# find_dir_size
# =============================================================================


class FindDirSizeInput(BaseModel):
    """Input: root directory to scan (optional; default handled by tool)."""

    dir: Optional[Path | str] = None

    @field_serializer("dir")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class FindDirSizePayload(BaseModel):
    """Payload: directory file/size stats."""

    file_count: int
    total_size_bytes: int
    total_size_mb: float


class FindDirSizeResult(ToolOutput):
    """Output: size stats under result."""

    result: Optional[FindDirSizePayload] = None


# =============================================================================
# find_dir_structure
# =============================================================================

NodeType = Literal["directory", "file"]


class FindDirStructureInput(BaseModel):
    """Input: build a directory file tree structure; optional root dir."""

    dir: Optional[Path | str] = None

    @field_serializer("dir")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class DirectoryNode(BaseModel):
    """
    Payload node: directory tree (recursive).
    Files: type='file', children=None.
    Dirs:  type='directory', children=[] (auto-normalized).
    """

    name: str
    type: NodeType
    children: Optional[List["DirectoryNode"]] = None

    @model_validator(mode="after")
    def _enforce_children_rules(self) -> "DirectoryNode":
        if self.type == "file" and self.children is not None:
            raise ValueError("Files must not include 'children'.")
        if self.type == "directory" and self.children is None:
            object.__setattr__(self, "children", [])
        return self


DirectoryNode.model_rebuild()  # forward refs


class FindDirStructureResult(ToolOutput):
    """Output: directory tree under result."""

    result: Optional[DirectoryNode] = None


# =============================================================================
# find_files_by_keyword
# =============================================================================


class FindFilesByKeywordsInput(BaseModel):
    """Input: keywords to match in filenames; optional root dir."""

    keywords: List[str]
    dir: Optional[Path | str] = None

    @field_serializer("dir")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class FindFilesByKeywordsPayload(BaseModel):
    """Payload: matched file paths."""

    files: List[str]


class FindFilesByKeywordsResult(ToolOutput):
    """Output: matched files under result."""

    result: Optional[FindFilesByKeywordsPayload] = None


# =============================================================================
# list_project_files
# =============================================================================


class ListProjectFilesInput(BaseModel):
    """Input: list files in project; optional root dir."""

    dir: Optional[Path | str] = None

    @field_serializer("dir")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class ListProjectFilesPayload(BaseModel):
    """Payload: discovered file paths."""

    files: List[str]


class ListProjectFilesResult(ToolOutput):
    """Output: file listing under result."""

    result: Optional[ListProjectFilesPayload] = None


# =============================================================================
# llm_response_async
# =============================================================================


class LLMResponseAsyncInput(BaseModel):
    """Input: prompt sent to the LLM (pre-formatted upstream)."""

    prompt: str


class LLMResponseAsyncResult(ToolOutput):
    """Output: generated natural-language text under result."""

    result: Optional[str] = None


# =============================================================================
# locate_files
# =============================================================================


class LocateFilesInput(BaseModel):
    """Input: exact filename to locate across common root directory.

    Accepts either a string or a Path object for filename.
    Paths are automatically converted to strings.
    """

    filename: Union[str, Path]

    @field_serializer("filename")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class LocateFilesPayload(BaseModel):
    """
    Payload: resolved path (if found).

    Accepts either a string or Path for `path` and normalizes to string.
    """

    path: Optional[Union[Path, str]] = None

    @field_serializer("path")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class LocateFilesResult(ToolOutput):
    """Output: located file path under result."""

    result: Optional[LocateFilesPayload] = None


# =============================================================================
# make_virtualenv
# =============================================================================


class MakeVirtualEnvInput(BaseModel):
    """Input: target path where the venv should be created."""

    dir: Path | str

    @field_serializer("dir")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class MakeVirtualEnvPayload(BaseModel):
    """Payload: creation status and resolved path."""

    created: bool = True
    dir: Optional[Path | str] = None

    @field_serializer("dir")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class MakeVirtualEnvResult(ToolOutput):
    """Output: venv creation result under result."""

    result: Optional[MakeVirtualEnvPayload] = None


# =============================================================================
# read_files  (supports single str or list[str] for input)
# =============================================================================


class ReadFilesInput(BaseModel):
    """Input: one or more file paths to read."""

    files: List[str]  # always a list of strings after validation

    @field_validator("files", mode="before")
    @classmethod
    def coerce_to_str_list(cls, v: Union[str, Path, List[Union[str, Path]]]):
        # Accept single str/Path or list of str/Path -> return List[str]
        if isinstance(v, (str, Path)):
            return [str(v)]
        if isinstance(v, list):
            out = []
            for item in v:
                if isinstance(item, (str, Path)):
                    out.append(str(item))
                else:
                    raise TypeError("Each item in 'path' must be a str or Path.")
            if not out:
                raise ValueError("'path' must contain at least one item.")
            return out
        raise TypeError("'path' must be a str, Path, or list of those.")


class FileContent(BaseModel):
    """Payload item: (file, content) pair."""

    file: Path | str
    content: str

    @field_serializer("file")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class ReadFilesPayload(BaseModel):
    files: List[FileContent]


class ReadFilesResult(ToolOutput):
    """Output: list of file-content pairs from read_files tool."""

    result: Optional[ReadFilesPayload] = None


# =============================================================================
# retrieval_tool
# =============================================================================


class RetrievalToolInput(BaseModel):
    """Input: intent/keyword describing the secret to look for."""

    query: str


class RetrievalToolPayload(BaseModel):
    """Payload: matched keys/paths, redacted where appropriate."""

    matches: List[str] = Field(default_factory=list)


class RetrievalToolResult(ToolOutput):
    """Output: matches under result."""

    result: Optional[RetrievalToolPayload] = None


# =============================================================================
# seed_parser
# =============================================================================


class SeedParserInput(BaseModel):
    """Input: plaintext file to parse into key/value pairs."""

    file: Path | str

    @field_serializer("file")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class SeedParserPayload(BaseModel):
    """Payload: parsed key/value data."""

    parsed_data: Dict[str, Any]


class SeedParserResult(ToolOutput):
    """Output: parsed data under result."""

    result: Optional[SeedParserPayload] = None


# =============================================================================
# set_scope
# =============================================================================


class SetScopeInput(BaseModel):
    """
    Input: project root directory and optional branch subpaths (relative to root).
    - dir: absolute or relative path (optional)
    - branches: str or list[str] (coerced)
    """

    dir: Optional[str] = None
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
    """Payload: normalized scope."""

    dir: Optional[str] = None
    branches: List[str] = Field(default_factory=list)


class SetScopeResult(ToolOutput):
    """Output: normalized scope under result."""

    result: Optional[SetScopePayload] = None


# =============================================================================
# select_files
# =============================================================================


class SortBy(str, Enum):
    filename = "name"
    path = "path"
    mtime = "mtime"
    ctime = "ctime"
    atime = "atime"


class Order(str, Enum):
    asc = "asc"
    desc = "desc"


class SelectFilesInput(BaseModel):
    """
    Input model for select_files tool. All incoming file-like values are coerced to List[Path].
    """

    files: List[Path] = Field(default_factory=list)
    include_ext: List[str] = Field(default_factory=list)
    exclude_ext: List[str] = Field(default_factory=list)
    include_name: List[str] = Field(default_factory=list)
    exclude_name: List[str] = Field(default_factory=list)
    sort_by: Optional[SortBy] = None
    order: Order = Order.asc
    offset: int = 0
    limit: Optional[int] = None
    tail: bool = False
    hard_cap: int = 100

    @field_validator("files", mode="before")
    @classmethod
    def _coerce_files_to_paths(cls, v: Any) -> List[Path]:
        """
        Accept str | Path | Iterable[str|Path] and convert to List[Path].
        Expands '~' and strips whitespace; does NOT resolve() to avoid FS hits.
        """

        def to_path(x: Any) -> Optional[Path]:
            if isinstance(x, Path):
                return x
            if isinstance(x, str) and x.strip():
                return Path(x.strip()).expanduser()
            return None

        if v is None:
            return []
        if isinstance(v, (str, Path)):
            p = to_path(v)
            return [p] if p else []
        if isinstance(v, Iterable):
            out: List[Path] = []
            for item in v:
                p = to_path(item)
                if p:
                    out.append(p)
            return out
        raise TypeError("files must be a path, string, or iterable of those")

    @field_serializer("files")
    def _files_to_str(self, v: List[Path]) -> List[str]:
        return [str(p) for p in v]


class SelectFilesApplied(BaseModel):
    """Applied/normalized arguments snapshot."""

    tail: bool
    offset: int
    limit: Optional[int]
    sort_by: Optional[str]
    order: str
    include_ext: List[str]
    exclude_ext: List[str]
    include_name: List[str]
    exclude_name: List[str]
    hard_cap: int


class SelectFilesPayload(BaseModel):
    """
    Payload returned by select_files under 'result'.
    Paths are serialized back to strings.
    """

    files: List[Path]
    total: int
    selected: int
    applied: SelectFilesApplied

    @field_serializer("files")
    def _files_to_str(self, v: List[Path]) -> List[str]:
        return [str(p) for p in v]


class SelectFilesResult(BaseModel):
    """
    Standard tool output envelope for select_files.
    """

    status: StepStatus
    message: Optional[str] = None
    result: Optional[SelectFilesPayload] = None


# =============================================================================
# summarize_config
# =============================================================================


class SummarizeConfigInput(BaseModel):
    """Input: summarize config files in root directory."""

    dir: Optional[Path | str] = None

    @field_serializer("dir")
    def _path_as_str(self, v: Path | None):
        return str(v) if v is not None else None


class SummarizeConfigResult(ToolOutput):
    """Output: plain-text configuration summary under result."""

    result: Optional[str] = None


# =============================================================================
# summarize_files
# =============================================================================


class SummarizeFilesInput(BaseModel):
    """Input: list of config file paths to summarize."""

    files: List[str]


class SummarizeFilesResult(ToolOutput):
    """Output: plain-text per-file summary under result."""

    result: Optional[str] = None
