"""Centralised application exceptions and handlers.

All HTTP errors raised by services/repositories should be typed
subclasses of `AppError` defined here. Handlers convert them — and any
unhandled exception — into the project's standard error response shape,
never leaking internal details to clients.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger
from utils.response import error_response

logger = get_logger(__name__)


class AppError(StarletteHTTPException):
    """Base class for all application HTTP errors.

    Subclass this with a fixed `status_code` and a client-safe message.
    """

    def __init__(
        self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST
    ) -> None:
        super().__init__(status_code=status_code, detail=detail)


class BadRequestError(AppError):
    def __init__(self, detail: str = "Bad request") -> None:
        super().__init__(detail, status.HTTP_400_BAD_REQUEST)


class UnauthorizedError(AppError):
    def __init__(self, detail: str = "Not authenticated") -> None:
        super().__init__(detail, status.HTTP_401_UNAUTHORIZED)


class ForbiddenError(AppError):
    def __init__(self, detail: str = "Not permitted") -> None:
        super().__init__(detail, status.HTTP_403_FORBIDDEN)


class NotFoundError(AppError):
    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(detail, status.HTTP_404_NOT_FOUND)


class ConflictError(AppError):
    def __init__(self, detail: str = "Resource already exists") -> None:
        super().__init__(detail, status.HTTP_409_CONFLICT)


class SessionNotFoundError(NotFoundError):
    def __init__(self, session_id: object) -> None:
        super().__init__(f"Session '{session_id}' not found")


class UnprocessableEntityError(AppError):
    def __init__(self, detail: str = "Unprocessable entity") -> None:
        super().__init__(detail, status.HTTP_422_UNPROCESSABLE_CONTENT)


class TooManyRequestsError(AppError):
    def __init__(self, detail: str = "Too many requests") -> None:
        super().__init__(detail, status.HTTP_429_TOO_MANY_REQUESTS)


class ServiceUnavailableError(AppError):
    def __init__(self, detail: str = "Service unavailable") -> None:
        super().__init__(detail, status.HTTP_503_SERVICE_UNAVAILABLE)


async def _http_exception_handler(
    _: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Render any `HTTPException` (incl. `AppError`) in the standard shape."""

    return error_response(message=str(exc.detail), status_code=exc.status_code)


async def _validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a 422 with field-level details but no internal traceback."""

    return error_response(
        message="Validation failed",
        data={"errors": exc.errors()},
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Log the full trace, return a safe generic 500 to the client."""

    logger.exception("Unhandled exception: %s", exc)
    return error_response(
        message="Internal server error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire all exception handlers onto the app. Call from the factory."""

    # Starlette types handler args as bare `Exception`; our narrowed signatures
    # are correct at runtime via its exception mapping, hence the ignores below.
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_exception_handler)


__all__ = [
    "AppError",
    "BadRequestError",
    "UnauthorizedError",
    "ForbiddenError",
    "NotFoundError",
    "ConflictError",
    "SessionNotFoundError",
    "UnprocessableEntityError",
    "TooManyRequestsError",
    "ServiceUnavailableError",
    "register_exception_handlers",
]
