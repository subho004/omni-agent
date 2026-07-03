"""Repository for `AgentSession` persistence."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import resolve_model, resolve_thinking_level
from app.models.agent_session import AgentSession
from app.models.artifact import Artifact
from app.models.ledger_entry import LedgerEntry
from app.models.message import Message
from app.models.plan_node import PlanNode
from app.models.run_event import RunEvent


class AgentSessionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        title: str,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> AgentSession:
        session = AgentSession(
            title=title,
            model=resolve_model(model),
            thinking_level=resolve_thinking_level(thinking_level),
        )
        self._db.add(session)
        await self._db.commit()
        await self._db.refresh(session)
        return session

    async def update_model(
        self, session_id: UUID, model: str | None, thinking_level: str | None
    ) -> None:
        """Persist a model/thinking change; ignores None (leave as-is)."""

        session = await self._db.get(AgentSession, session_id)
        if session is None:
            return
        if model is not None:
            session.model = resolve_model(model)
        if thinking_level is not None:
            session.thinking_level = resolve_thinking_level(thinking_level)
        await self._db.commit()

    async def find_by_id(self, session_id: UUID) -> AgentSession | None:
        return await self._db.get(AgentSession, session_id)

    async def list_page(
        self, page: int, page_size: int
    ) -> tuple[list[AgentSession], int]:
        total = (
            await self._db.execute(select(func.count(AgentSession.id)))
        ).scalar_one()
        rows = await self._db.execute(
            select(AgentSession)
            .order_by(AgentSession.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list(rows.scalars().all()), total

    async def add_tokens_used(self, session_id: UUID, tokens: int) -> None:
        session = await self._db.get(AgentSession, session_id)
        if session is not None:
            session.tokens_used += tokens
            await self._db.commit()

    async def delete(self, session_id: UUID) -> bool:
        """Delete a session and all its rows. Returns False if it didn't exist.

        Child tables have no ON DELETE CASCADE, so we clear them explicitly
        before removing the session row.
        """

        session = await self._db.get(AgentSession, session_id)
        if session is None:
            return False
        for model in (Message, PlanNode, Artifact, LedgerEntry, RunEvent):
            await self._db.execute(
                delete(model).where(model.session_id == session_id)
            )
        await self._db.delete(session)
        await self._db.commit()
        return True
