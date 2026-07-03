"""Repository for `PlanNode` persistence."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.plan_node import PlanNode


class PlanNodeRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        session_id: UUID,
        step_number: int,
        title: str,
        description: str,
        depends_on: list[int],
        turn: int = 1,
    ) -> PlanNode:
        node = PlanNode(
            session_id=session_id,
            turn=turn,
            step_number=step_number,
            title=title,
            description=description,
            depends_on=depends_on,
        )
        self._db.add(node)
        await self._db.commit()
        await self._db.refresh(node)
        return node

    async def list_by_session(self, session_id: UUID) -> list[PlanNode]:
        rows = await self._db.execute(
            select(PlanNode)
            .where(PlanNode.session_id == session_id)
            .order_by(PlanNode.turn.asc(), PlanNode.step_number.asc())
        )
        return list(rows.scalars().all())

    async def list_by_turn(self, session_id: UUID, turn: int) -> list[PlanNode]:
        """Nodes for one research turn (the scheduling/synthesis scope)."""

        rows = await self._db.execute(
            select(PlanNode)
            .where(PlanNode.session_id == session_id, PlanNode.turn == turn)
            .order_by(PlanNode.step_number.asc())
        )
        return list(rows.scalars().all())

    async def latest_turn(self, session_id: UUID) -> int:
        """Highest turn number in the session, or 0 if none exist yet."""

        current = (
            await self._db.execute(
                select(func.max(PlanNode.turn)).where(
                    PlanNode.session_id == session_id
                )
            )
        ).scalar()
        return current or 0

    async def next_step_number(self, session_id: UUID) -> int:
        current = (
            await self._db.execute(
                select(func.max(PlanNode.step_number)).where(
                    PlanNode.session_id == session_id
                )
            )
        ).scalar()
        return (current or 0) + 1

    async def set_status(self, node_id: UUID, status: str) -> None:
        node = await self._db.get(PlanNode, node_id)
        if node is not None:
            node.status = status
            await self._db.commit()

    async def set_result(self, node_id: UUID, status: str, result: str) -> None:
        node = await self._db.get(PlanNode, node_id)
        if node is not None:
            node.status = status
            node.result = result
            await self._db.commit()
