"""Repository for `Artifact` persistence."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.artifact import Artifact


class ArtifactRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self, session_id: UUID, kind: str, name: str, uri: str, summary: str = ""
    ) -> Artifact:
        artifact = Artifact(
            session_id=session_id, kind=kind, name=name, uri=uri, summary=summary
        )
        self._db.add(artifact)
        await self._db.commit()
        await self._db.refresh(artifact)
        return artifact

    async def find_by_id(self, artifact_id: UUID) -> Artifact | None:
        return await self._db.get(Artifact, artifact_id)

    async def list_by_session(self, session_id: UUID) -> list[Artifact]:
        rows = await self._db.execute(
            select(Artifact)
            .where(Artifact.session_id == session_id)
            .order_by(Artifact.created_at.asc())
        )
        return list(rows.scalars().all())
