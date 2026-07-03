"""API response wrapper utilities.

Provide a generic `APIResponse` Pydantic model that can wrap any
response payload. Also include small helper factory functions for
common success/error responses.

This module is intended for use across the application to provide a
consistent response schema.
"""

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper.

    Fields:
    - status: 'success' or 'error'
    - message: optional human-readable message
    - data: optional payload (object)
    """

    status: str = "success"
    message: Optional[str] = None
    data: Optional[T] = None


def success(data: Optional[T] = None, message: Optional[str] = None) -> APIResponse[T]:
    """Create a success `APIResponse`.

    Args:
        data: Optional payload to include.
        message: Optional human-readable message.

    Returns:
        An `APIResponse` with `status` set to "success".
    """

    return APIResponse(status="success", message=message, data=data)


def error(message: str, data: Optional[T] = None) -> APIResponse[T]:
    """Create an error `APIResponse`.

    Args:
        message: Human-readable error message.
        data: Optional payload (e.g., validation details).

    Returns:
        An `APIResponse` with `status` set to "error".
    """

    return APIResponse(status="error", message=message, data=data)


def to_http_response(obj: APIResponse, status_code: int = 200) -> JSONResponse:
    """Convert an `APIResponse` into a FastAPI `JSONResponse`.

    This helper uses `jsonable_encoder` so Pydantic models are serialized
    correctly and returns a consistent JSON structure for the API.
    """

    payload = jsonable_encoder(obj)
    return JSONResponse(content=payload, status_code=status_code)


def success_response(
    data: Optional[T] = None, message: Optional[str] = None, status_code: int = 200
) -> JSONResponse:
    """Create and return a JSONResponse for a successful result."""

    return to_http_response(
        success(data=data, message=message), status_code=status_code
    )


def error_response(
    message: str, data: Optional[T] = None, status_code: int = 400
) -> JSONResponse:
    """Create and return a JSONResponse for an error result."""

    return to_http_response(error(message=message, data=data), status_code=status_code)


__all__ = ["APIResponse", "success", "error"]
