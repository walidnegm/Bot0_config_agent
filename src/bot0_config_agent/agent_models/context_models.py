"""bot0_config_agewnt/agent_models/context_models.py"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Literal, Optional, Sequence
from pydantic import BaseModel, Field, ConfigDict


logger = logging.getLogger(__name__)

Role = Literal["system", "user", "tool"]


class Message(BaseModel):
    role: Role
    content: str = Field(..., description="tba")
    name: Optional[str] = None
    at: datetime = Field(default_factory=datetime.now)
