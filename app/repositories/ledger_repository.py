"""Repository for `LedgerEntry` persistence (LLM token usage)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ledger_entry import LedgerEntry


class LedgerRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self, session_id: UUID, model: str, input_tokens: int, output_tokens: int
    ) -> LedgerEntry:
        entry = LedgerEntry(
            session_id=session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._db.add(entry)
        await self._db.commit()
        await self._db.refresh(entry)
        return entry
