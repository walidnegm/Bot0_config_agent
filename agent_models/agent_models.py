"""
File: agent_response_models.py
Last updated: 2025-07-30

Pydantic models for validating and structuring LLM/agent responses for
a tool-calling agent.

This module standardizes outputs for agent plans, tool results, text, code, and
generic JSON responses. It is designed for flexibility, robust validation, and
extensibility as new agent capabilities emerge.

Model hierarchy:
    - BaseResponseModel: Base with status/message (all responses inherit).
    - TextResponse / CodeResponse: For natural language or code snippet
        agent outputs.
    - JSONResponse: For any generic JSON result.
    - ToolCall / ToolPlan / ToolResult / ToolResults: For structured
        tool-calling, planning,
        and results.
    - IntentClassificationResponse: (Optional) For intent/meta outputs
        from classifier heads.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator, ConfigDict


# --- 1. Base Response (status/message for all outputs) ---
class BaseResponseModel(BaseModel):
    """
    Base model providing common fields for all agent responses.

    Attributes:
        status (str): Indicates success or failure of the agent response.
            Default is "success".
        message (Optional[str]): Optional field for feedback, error reporting,
            or status messages.

    Config:
        arbitrary_types_allowed (bool): Permits non-standard types in subclass models
            (e.g., DataFrames).
    """

    status: str = "success"
    message: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --- 2. Text and Code Agent Responses ---
class TextResponse(BaseResponseModel):
    """
    Model for plain text agent responses.

    Attributes:
        content (str): Natural language content returned by the agent.
        Inherits status and message from BaseResponseModel.

    Example:
        {
            "status": "success",
            "message": "Text response processed.",
            "content": "This is the plain text content."
        }
    """

    content: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Text response processed.",
                "content": "This is the plain text content.",
            }
        }
    )


class CodeResponse(BaseResponseModel):
    """
    Model for responses containing code blocks or scripts.

    Attributes:
        code (str): The returned code snippet as a string.
        Inherits status and message from BaseResponseModel.

    Example:
        {
            "status": "success",
            "message": "Code response processed.",
            "code": "print('Hello, world!')"
        }
    """

    code: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Code response processed.",
                "code": "print('Hello, world!')",
            }
        }
    )


# --- 3. Generic JSON Response (for untyped or flexible outputs) ---
class JSONResponse(BaseResponseModel):
    """
    General-purpose model for agent outputs that are valid JSON.

    Attributes:
        data (Union[Dict[str, Any], List[Any]]): Arbitrary JSON content returned
            by the agent/tool.

    Config:
        arbitrary_types_allowed (bool): Allows lists or dicts as data.

    Example:
        {
            "status": "success",
            "message": "JSON response processed.",
            "data": [{"field": "value"}]
        }
    """

    data: Union[Dict[str, Any], List[Any]]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "JSON response processed.",
                "data": [{"field": "value"}],
            }
        },
    )


# --- 4. Tool Calling: Plan, Call, Results ---
class ToolCall(BaseModel):
    """
    Represents a single tool selection/action in an agent workflow.

    Attributes:
        tool (str): Name of the tool to invoke (must match the tool registry).
        params (Dict[str, Any]): Arguments/parameters for the tool.

    Validation:
        - Ensures 'tool' is a non-empty string.
        - Ensures 'params' is a dictionary.

    Example:
        {
            "tool": "list_project_files",
            "params": {"dir": "."}
        }
    """

    tool: str = Field(
        ..., description="Name of the tool to call, as registered in the tool registry."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters to pass to the tool function."
    )

    @model_validator(mode="after")
    def validate_tool_and_params(self):
        if not self.tool or not isinstance(self.tool, str):
            raise ValueError("Tool name must be a non-empty string.")
        if self.params is not None and not isinstance(self.params, dict):
            raise ValueError("Params must be a dictionary of arguments.")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"tool": "list_project_files", "params": {"dir": "."}}
        }
    )


class ToolChain(BaseModel):
    """
    Represents a multi-step tool-calling plan.

    Attributes:
        steps (List[ToolCall]): Ordered list of tool selections/actions.

    Validation:
        - Ensures at least one step is present.
        - Ensures every item is a valid ToolCall instance.

    Example:
        {
            "steps": [
                {"tool": "list_project_files", "params": {"dir": "."}},
                {"tool": "echo_message", "params": {"message": "<prev_output>"}}
            ]
        }
    """

    steps: List[ToolCall] = Field(
        ..., description="Ordered list of tool selections (multi-step plan)."
    )

    @model_validator(mode="after")
    def validate_steps(self):
        if not self.steps or not isinstance(self.steps, list):
            raise ValueError("ToolChain must have at least one ToolCall in 'steps'.")
        for step in self.steps:
            if not isinstance(step, ToolCall):
                raise ValueError("Each step must be a ToolCall instance.")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "steps": [
                    {"tool": "list_project_files", "params": {"dir": "."}},
                    {"tool": "echo_message", "params": {"message": "<prev_output>"}},
                ]
            }
        }
    )


# --- Intent Classification for Agent Routing ---
class IntentClassificationResponse(BaseResponseModel):
    """
    Model for agent intent classification outputs (optional, for future use).

    Attributes:
        intent (str): Detected or classified intent (e.g., "describe_project", "tool_call", etc).
        Inherits status/message from BaseResponseModel.

    Example:
        {
            "status": "success",
            "message": "Intent classified.",
            "intent": "describe_project"
        }
    """

    intent: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Intent classified.",
                "intent": "describe_project",
            }
        }
    )
