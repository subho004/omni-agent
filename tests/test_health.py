"""Sample test for the /health endpoint.

Serves as a template for writing further async API tests.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/", "/health"])
async def test_health_returns_ok(client: AsyncClient, path: str) -> None:
    response = await client.get(path)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["message"] == "healthy"
    assert body["data"] == {"status": "ok"}
