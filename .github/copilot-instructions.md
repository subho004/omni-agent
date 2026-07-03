# Backend Engineering Guide

> Governs all AI-assisted development on this FastAPI project (Copilot, Claude, Cursor, any LLM/editor).
> Follow these rules. When in doubt, match existing patterns.

---

## Engineering Principles

Apply these to every change. They override personal style.

- **KISS** — Simplest solution that works. If 200 lines can be 50, write 50.
- **YAGNI** — Build only what's asked. No speculative features, configs, or abstractions.
- **DRY** — Logic used in ≥ 2 places goes into `utils/` or a shared service. No copy-paste.
- **SoC** (Separation of Concerns) — Each layer has one job; never cross boundaries (see Layers).
- **SOLID**
  - **S**ingle Responsibility — one reason to change per class/function.
  - **O**pen/Closed — extend via new code, don't edit working code to add cases.
  - **L**iskov — subtypes must be drop-in replacements for their base.
  - **I**nterface Segregation — small focused interfaces over fat ones.
  - **D**ependency Inversion — depend on abstractions, inject via `Depends()`, never instantiate deps inline.

---

## AI Behavior

- **Think first.** State assumptions; if uncertain, ask. Present alternatives instead of guessing.
- **Surgical changes.** Touch only what the task requires. Don't refactor or reformat adjacent code. Flag smells — don't silently fix them.
- **Goal-driven.** Turn vague tasks into verifiable ones ("fix the bug" → "reproduce in a test, then fix"). Loop until verified.

---

## Stack

Python 3.14.2 (via `uv`) · FastAPI · Uvicorn · SQLAlchemy 2.x async + asyncpg (PostgreSQL) · Pydantic v2 + pydantic-settings · Alembic · pytest/pytest-asyncio/httpx · Ruff · Mypy · pre-commit · Docker (slim + uv)

---

## Commands

> Python **3.14.2**, managed with [`uv`](https://docs.astral.sh/uv/). `uv run` executes inside the project env automatically — no manual activation needed.

```bash
uv venv --python 3.14.2           # one-time: create env
uv pip install -r requirements.txt

uv run uvicorn main:app --reload  # run (entrypoint is main:app at repo root)
uv run pytest -v                  # test
uv run ruff check . && uv run mypy --explicit-package-bases .   # lint + type check
pre-commit install                # one-time: enable git hooks (ruff + mypy on commit)
docker compose up --build         # run via Docker (slim image, uv install)
```

---

## Project Structure & Layers

```
main.py              # entrypoint: create_app() factory + uvicorn runner
app/
├── api/v1/          # routers — health.py, router.py (aggregator); add feature routers here
├── core/            # config.py (pydantic-settings), exceptions.py, logging.py (JSON)
├── db/              # async engine / session / Base
├── services/        # business logic / use-case orchestration
├── repositories/    # all DB access & ORM queries
├── schemas/         # Pydantic DTOs (request/response)
└── models/          # SQLAlchemy ORM models
utils/               # pure, stateless helpers (response.py, …)
tests/               # pytest + httpx AsyncClient (conftest.py)
```

> `app/` uses namespace packages (no `__init__.py`) — run mypy with `--explicit-package-bases`.

| Layer           | Allowed                                | Forbidden                          |
| --------------- | -------------------------------------- | ---------------------------------- |
| `routes/`       | parse input, call service, respond     | DB queries, business logic         |
| `services/`     | orchestrate use-cases, call repos      | HTTP types (`Request`/`Response`)  |
| `repositories/` | DB access, ORM queries                 | business rules, HTTP concerns      |
| `schemas/`      | Pydantic I/O models                    | DB models, service imports         |
| `utils/`        | pure stateless helpers                 | state, DB, HTTP                    |

---

## Code Style

- Full type annotations on every function/class. Avoid `Any` (if unavoidable, `# type: ignore` with reason).
- Descriptive names (`get_user_by_email`, not `get_user`). Functions ≤ ~30 lines, single responsibility.
- No `print()` — use the structured logger. No bare `except Exception` — handle or re-raise with context.
- Services & repositories are **class-based** with deps injected via `__init__`. Prefer composition over inheritance.
- Helpers in `utils/` must be pure and grouped by concern (`utils/pagination.py`, `utils/hashing.py`).

```python
class UserService:
    def __init__(self, repo: UserRepository) -> None:
        self._repo = repo

    async def get_by_email(self, email: str) -> UserResponse:
        user = await self._repo.find_by_email(email)
        if not user:
            raise UserNotFoundError(email)
        return UserResponse.model_validate(user)
```

---

## API Standards

**Routes**: validate input → call service → return response.

```python
@router.get("/{user_id}", response_model=ApiResponse[UserResponse])
async def get_user(
    user_id: UUID,
    service: UserService = Depends(get_user_service),
) -> ApiResponse[UserResponse]:
    user = await service.get_by_id(user_id)
    return success_response(message="User retrieved", data=user)
```

**Responses**: always use the `utils.response` helpers (`success_response` / `error_response`) — never raw dicts.

```python
return success_response(message="Created", data=response_data)
return error_response(message="Not found", status_code=404)
```

**Status codes**: 201 created · 204 no content · 400 bad request · 401 unauthenticated · 403 forbidden · 404 not found · 409 conflict · 422 validation · 500 internal.

**Pagination**: all list endpoints must paginate via the shared utility — never return unbounded sets.

```python
async def list_users(page: int = 1, page_size: int = 20) -> PaginatedResponse[UserResponse]: ...
```

**URL shape — never end a path with an id/uid.** End with a resource word so requests are scannable in the network tab (you can tell what was called without decoding the id). Put the id in the middle, followed by the sub-resource or action.

```python
# Good — readable in network tab
@router.get("/users/{user_id}/profile")
@router.get("/orders/{order_id}/items")
@router.post("/users/{user_id}/deactivate")

# Bad — trailing id, opaque in network tab
@router.get("/users/{user_id}")
@router.get("/orders/{order_id}")
```

For a single resource that has no natural sub-word, append the resource name: `/users/{user_id}/details` instead of `/users/{user_id}`.

---

## Error Handling

All HTTP errors live in `app/core/exceptions.py`. They subclass `AppError` (a typed `HTTPException`) and are converted to the standard response shape by handlers registered via `register_exception_handlers(app)` in the app factory. Never raise raw `HTTPException` or expose internal details to clients.

**Available exceptions** (raise these from services/repositories — never build status codes by hand):

| Exception                  | Code |
| -------------------------- | ---- |
| `BadRequestError`          | 400  |
| `UnauthorizedError`        | 401  |
| `ForbiddenError`           | 403  |
| `NotFoundError`            | 404  |
| `ConflictError`            | 409  |
| `UnprocessableEntityError` | 422  |
| `TooManyRequestsError`     | 429  |
| `ServiceUnavailableError`  | 503  |

For a domain-specific error, subclass the closest one with a fixed, client-safe message:

```python
# app/core/exceptions.py
class UserNotFoundError(NotFoundError):
    def __init__(self, user_id: UUID | str) -> None:
        super().__init__(f"User '{user_id}' not found")
```

```python
# in a service — log internals, raise a safe typed error
try:
    result = await self._repo.create(data)
except IntegrityError as exc:
    logger.exception("DB constraint violation")
    raise ConflictError("User already exists") from exc
```

The registered handlers cover: any `AppError`/`HTTPException` → standard shape; `RequestValidationError` → 422 with field errors; any unhandled `Exception` → logged with full trace, returned as a generic 500. Don't add per-route `try/except` for these — let them propagate.

---

## Database

- `async/await` for all DB ops — no sync calls. Sessions only via `Depends()`, never created inside services.
- ORM models use `Mapped[T]` columns (SQLAlchemy 2.x). UUID primary keys unless schema differs.
- Declare relationships, constraints, and indexes explicitly.
- Avoid N+1 with `selectinload`/`joinedload`. Prefer bulk ops over per-row loops.

```python
class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
```

---

## Configuration

Never hardcode secrets, URLs, or env-specific values. Use the single `settings` instance — import only via `from app.core.config import settings`.

Settings use **`pydantic-settings`** (`BaseSettings` + `SettingsConfigDict`), loaded from the environment and an optional `.env`:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    database_url: str = ""
    debug: bool = False

settings = Settings()
```

For `list[str]` fields fed by a comma-separated env var, annotate with `NoDecode` and split in a `@field_validator(mode="before")` — pydantic-settings otherwise JSON-decodes the value first.

---

## Logging

Use `core/logging.py`. Log start/end of important ops and all failures with stack traces.

- Never log secrets, tokens, passwords, or PII.
- Use `%s`-style lazy formatting, not f-strings.
- Use `logger.exception()` (not `.error()`) to auto-attach the stack trace.

```python
async def run_sync_job() -> None:
    start = time.perf_counter()
    logger.info("Sync job started")
    try:
        await _do_work()
        logger.info("Sync job completed in %.3fs", time.perf_counter() - start)
    except Exception:
        logger.exception("Sync job failed after %.3fs", time.perf_counter() - start)
        raise
```

---

## Security

- All SQL through the ORM — no raw string queries.
- Validate all external input through Pydantic before it reaches services.
- Secrets only via env/`settings`. Never log or expose tokens, passwords, or PII.
- All `HTTPException` messages must be safe for public clients.

---

## Naming

| Kind                | Style               | Example                       |
| ------------------- | ------------------- | ----------------------------- |
| Variables/functions | `snake_case`        | `get_user_by_email()`         |
| Classes             | `PascalCase`        | `UserService`                 |
| Constants           | `UPPER_SNAKE`       | `MAX_RETRY_COUNT`             |
| Private attrs       | `_prefix`           | `self._repo`                  |
| Python files        | `snake_case`        | `user_service.py`             |
| Route files         | plural noun         | `users.py`                    |
| Service/repo/schema | singular noun       | `user_service.py`             |
| Test files          | `test_` prefix      | `test_user_service.py`        |
| Tables              | `snake_case` plural | `users`, `order_items`        |
| Columns             | `snake_case`        | `created_at`, `is_active`     |
| Index / unique / FK | `ix_` / `uq_` / `fk_` | `ix_users_email`            |
| Primary key         | always `id`         | `id`                          |
| URL segments        | lowercase kebab     | `/order-items/`, `/users/{user_id}/profile` |

Suffix Pydantic schemas by purpose: `UserCreate`, `UserUpdate`, `UserResponse`, `UserInDB`.

**Imports**: stdlib → third-party → internal (`api.*`). Enforced by Ruff isort rules.

---

## Testing

> Tests are optional — write them only when explicitly asked.

When requested: pytest + pytest-asyncio · unit tests mock repositories · integration tests use a real test DB · ≥ 80% coverage on services/repos · override deps with `app.dependency_overrides`, never patch globals.

**Leave no stale data.** Every test must clean up whatever it creates — DB rows, files, cache/queue entries, external resources — so the suite is repeatable and order-independent. Do teardown in fixtures (`yield` + cleanup, or transaction rollback per test), not ad-hoc at the end of a test body, so it runs even when assertions fail. After a run, the test DB and environment must be in the same state as before it.

---

## Scripts

- Constants at top instead of CLI args (unless asked). Add dynamic `sys.path` so it runs from any cwd.
- Log start/end/elapsed. Support `python path/to/script.py`. Run inside `.venv`.

```python
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
```

---

## Validation Checklist (after every change)

```bash
uv run python -m py_compile path/to/file.py        # 1. syntax/imports
uv run ruff check .                                 # 2. lint
uv run mypy --explicit-package-bases .              # 3. types (app/ is a namespace package)
```

Steps 1–3 must pass cleanly before a task is complete — fix root causes, don't suppress warnings.
Tests/coverage (`uv run pytest -v`, `uv run pytest --cov=app`) only when explicitly requested.
