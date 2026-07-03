"""Repository for `Message` persistence."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message


class MessageRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, session_id: UUID, role: str, content: str) -> Message:
        message = Message(session_id=session_id, role=role, content=content)
        self._db.add(message)
        await self._db.commit()
        await self._db.refresh(message)
        return message

    async def list_by_session(self, session_id: UUID) -> list[Message]:
        rows = await self._db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
        )
        return list(rows.scalars().all())

    async def list_page(
        self, session_id: UUID, page: int, page_size: int
    ) -> tuple[list[Message], int]:
        total = (
            await self._db.execute(
                select(func.count(Message.id)).where(Message.session_id == session_id)
            )
        ).scalar_one()
        rows = await self._db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list(rows.scalars().all()), total
