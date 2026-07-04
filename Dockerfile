# syntax=docker/dockerfile:1
FROM python:3.14.2-slim

# uv: fast, reproducible dependency installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    # Shared, world-readable browser location so the non-root runtime user can
    # launch the Chromium that crawl_url / browser_use need.
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers

WORKDIR /app

# 1) Python deps first, for better layer caching.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# 2) Chromium + its OS libraries for the headless-browser tools: crawl4ai runs
#    via playwright, browser_use via patchright. `--with-deps` installs the apt
#    packages Chromium needs; patchright shares those same system libs and only
#    downloads its browser build. Both honor PLAYWRIGHT_BROWSERS_PATH.
RUN playwright install --with-deps chromium \
    && patchright install chromium \
    && chmod -R a+rx "$PLAYWRIGHT_BROWSERS_PATH"

# 3) Application source.
COPY . .

# 4) Non-root runtime user. Creating /app/data here (owned by appuser) means a
#    named volume mounted over it inherits these permissions, so uploads, parsed
#    docs, the embedding cache, and the SQLite DB stay writable.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Bind to 0.0.0.0 so the port is reachable from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
