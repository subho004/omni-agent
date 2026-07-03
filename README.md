# Backend Setup

This repository is a draft FastAPI backend setup intended to be used as a reusable template.

Uses **Python 3.14.2** and [`uv`](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

## Prerequisites

Install `uv` (see the [docs](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

1. Create a virtual environment with the pinned Python version:

```bash
uv venv --python 3.14.2
```

2. Activate it:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
```

## Run the app

Start the FastAPI server with Uvicorn:

```bash
uv run uvicorn main:app --reload
```

Then hit the health check at `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/health`.

> If your FastAPI app is located in a different module, update `main:app` accordingly.

## Test

```bash
uv run pytest -v
```

## Docker

Build and run the API with Docker Compose:

```bash
docker compose up --build
```

The image uses the `python:3.14.2-slim` base and installs dependencies with `uv`.

## Notes

- `uv run` executes commands inside the project environment automatically — no manual activation needed.
- For production deployments, replace `--reload` with a production-ready configuration.
