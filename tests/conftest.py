"""Shared pytest fixtures.

Provides an async HTTP client wired to the FastAPI app via an in-process
ASGI transport — no network or running server required. The database is
pointed at a per-run temporary SQLite file (set BEFORE importing `main`
so the module-level engine binds to it) and removed afterwards.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

_TMP_DIR = tempfile.mkdtemp(prefix="harness-tests-")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DIR}/test.db"
os.environ["DATA_DIR"] = str(Path(_TMP_DIR) / "data")

import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.db.database import engine, init_db  # noqa: E402
from main import app  # noqa: E402


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    # Drop all tables so every test starts from a clean database.
    from app.db.database import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
