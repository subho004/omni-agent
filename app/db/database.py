"""Async database engine, session factory, and declarative base.

Uses SQLite via ``aiosqlite`` (per docs/idea.md: everything in-memory /
local SQLite). The database URL comes from settings; when unset it
defaults to a local file under ``data/``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

DEFAULT_SQLITE_URL = "sqlite+aiosqlite:///./data/harness.db"


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _database_url() -> str:
    return settings.database_url or DEFAULT_SQLITE_URL


def _ensure_sqlite_dir(url: str) -> None:
    """Create the parent directory for a file-backed SQLite database."""

    prefix = "sqlite+aiosqlite:///"
    if url.startswith(prefix) and ":memory:" not in url:
        Path(url.removeprefix(prefix)).parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_dir(_database_url())

# `timeout` sets SQLite's busy timeout so concurrent writers (parallel
# sub-agents share one session's UUID but use separate connections) wait for
# the lock instead of raising "database is locked".
_connect_args = (
    {"timeout": 30} if _database_url().startswith("sqlite") else {}
)

engine = create_async_engine(
    _database_url(), echo=False, connect_args=_connect_args
)

async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    """Create all tables. Called once at application startup."""

    # Import model modules so they register on Base.metadata before create_all
    # (app/ is a namespace package, so each module is imported explicitly).
    from app.models import (  # noqa: F401
        agent_session,
        artifact,
        ledger_entry,
        message,
        plan_node,
        run_event,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_apply_lightweight_migrations)


def _apply_lightweight_migrations(conn: object) -> None:
    """Additive schema tweaks for pre-existing SQLite databases.

    We use ``create_all`` (no Alembic), which never alters existing tables,
    so a column added to a model after a DB already exists must be backfilled
    here. Each step is idempotent (checked against the live schema).
    """

    from sqlalchemy import inspect, text
    from sqlalchemy.engine import Connection

    assert isinstance(conn, Connection)
    inspector = inspect(conn)
    tables = set(inspector.get_table_names())
    if "plan_nodes" in tables:
        columns = {col["name"] for col in inspector.get_columns("plan_nodes")}
        if "turn" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE plan_nodes ADD COLUMN turn INTEGER NOT NULL "
                    "DEFAULT 1"
                )
            )
    if "agent_sessions" in tables:
        from app.core.models import DEFAULT_MODEL, DEFAULT_THINKING_LEVEL

        columns = {col["name"] for col in inspector.get_columns("agent_sessions")}
        if "model" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE agent_sessions ADD COLUMN model VARCHAR(64) "
                    f"NOT NULL DEFAULT '{DEFAULT_MODEL}'"
                )
            )
        if "thinking_level" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE agent_sessions ADD COLUMN thinking_level "
                    f"VARCHAR(16) NOT NULL DEFAULT '{DEFAULT_THINKING_LEVEL}'"
                )
            )
        for col in ("input_tokens_used", "output_tokens_used"):
            if col not in columns:
                conn.execute(
                    text(
                        f"ALTER TABLE agent_sessions ADD COLUMN {col} "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                )


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped async session."""

    async with async_session_factory() as session:
        yield session


__all__ = [
    "Base",
    "engine",
    "async_session_factory",
    "init_db",
    "get_db_session",
]
