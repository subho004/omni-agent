"""ORM model for a persisted run event (activity trace).

Every meaningful event the orchestrator emits during a run (plan_created,
node_started, tool, reflection, evaluator, node_completed, …) is stored so the
activity trace survives beyond the live SSE stream — letting the UI replay it
when a session is reopened and include it in a full export.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_sessions.id"), index=True, nullable=False
    )
    # Monotonic sequence within a session, so events replay in emit order.
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    # The event payload minus its "type" (step_number, tool, reason, …).
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
