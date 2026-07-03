"""Health check routes.

Exposes liveness endpoints on both the base path (`/`) and `/health`
so load balancers, container probes, and humans all have a target.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from utils.response import success_response

router = APIRouter(tags=["health"])


@router.get("/")
@router.get("/health")
async def health() -> JSONResponse:
    """Return service liveness status."""

    return success_response(data={"status": "ok"}, message="healthy")
