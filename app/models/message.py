"""ORM model for a message within an agent session.

Stores user/assistant turns and tool-call traces (`role` = "tool") so the
UI can replay the agent's thinking and the executor can rebuild context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_sessions.id"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Python-side default keeps microsecond precision on SQLite so
    # same-second messages within a turn stay correctly ordered.
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
