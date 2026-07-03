"""Pydantic DTOs for sessions, messages, chat, and artifacts."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SessionCreate(BaseModel):
    title: str = Field(default="New session", max_length=255)
    model: str | None = Field(default=None, description="Gemini model id.")
    thinking_level: str | None = Field(
        default=None, description="Reasoning depth: low | medium | high."
    )


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    status: str
    tokens_used: int
    model: str
    thinking_level: str
    created_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    role: str
    content: str
    created_at: datetime


class ArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    kind: str
    name: str
    summary: str
    created_at: datetime


class ModelChoice(BaseModel):
    """Optional per-request model/thinking override (persisted on the session)."""

    model: str | None = None
    thinking_level: str | None = None


class ChatRequest(ModelChoice):
    message: str = Field(min_length=1)


class ToolCallTrace(BaseModel):
    tool: str
    args: dict[str, object]
    result_summary: str


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    tool_calls: list[ToolCallTrace]
    input_tokens: int
    output_tokens: int


class PaginatedMessages(BaseModel):
    items: list[MessageResponse]
    page: int
    page_size: int
    total: int


class PaginatedSessions(BaseModel):
    items: list[SessionResponse]
    page: int
    page_size: int
    total: int
