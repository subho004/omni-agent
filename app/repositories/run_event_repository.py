"""Repository for `RunEvent` persistence (the activity trace)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.run_event import RunEvent


class RunEventRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def append(
        self, session_id: UUID, event_type: str, data: dict
    ) -> None:
        """Store one event, assigning the next per-session sequence number."""

        current = (
            await self._db.execute(
                select(func.max(RunEvent.seq)).where(
                    RunEvent.session_id == session_id
                )
            )
        ).scalar()
        self._db.add(
            RunEvent(
                session_id=session_id,
                seq=(current or 0) + 1,
                type=event_type,
                data=data,
            )
        )
        await self._db.commit()

    async def list_by_session(self, session_id: UUID) -> list[RunEvent]:
        rows = await self._db.execute(
            select(RunEvent)
            .where(RunEvent.session_id == session_id)
            .order_by(RunEvent.seq.asc())
        )
        return list(rows.scalars().all())
