"""ORM model for an agent chat session."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.models import DEFAULT_MODEL, DEFAULT_THINKING_LEVEL
from app.db.database import Base


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(255), default="New session")
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    tokens_used: Mapped[int] = mapped_column(default=0)
    # Per-session model choice + reasoning depth (user-selectable in the UI).
    model: Mapped[str] = mapped_column(String(64), default=DEFAULT_MODEL)
    thinking_level: Mapped[str] = mapped_column(
        String(16), default=DEFAULT_THINKING_LEVEL
    )
    created_at: Mapped[datetime] = mapped_column(default=func.now())
