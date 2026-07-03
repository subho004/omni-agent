"""ORM model for a node in a session's research plan (DAG).

A plan is a set of nodes with `depends_on` edges (by `step_number`); the
orchestrator builds a NetworkX DAG from these and dispatches nodes whose
dependencies are complete (docs/implementation-plan.md Phases 7-9).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import JSON, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PlanNode(Base):
    __tablename__ = "plan_nodes"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_sessions.id"), index=True, nullable=False
    )
    # Which research turn this node belongs to. Each user query in a session
    # starts a new turn; step numbers restart at 1 per turn, so turns are
    # scheduled/synthesized independently while sharing the session's context.
    turn: Mapped[int] = mapped_column(Integer, default=1, index=True, nullable=False)
    # Per-turn step number used to express dependencies within the plan.
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    depends_on: Mapped[list[int]] = mapped_column(JSON, default=list)
    result: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
