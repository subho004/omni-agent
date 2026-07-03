# Omni-Agent

A Claude-Code-style research backend: it answers a question by **planning,
running tools, and self-correcting in a loop** — decomposing work into a plan,
executing steps as parallel sub-agents with real tools (web search, crawl,
browser automation, document parsing, code/shell exec), then evaluating,
replanning, and reflecting until the answer is genuinely sufficient. Progress
and the final Markdown answer stream to a minimal web UI with multi-session
chat history.

See [`.github/REPO_MAP.md`](.github/REPO_MAP.md) for the architecture and
[`.github/copilot-instructions.md`](.github/copilot-instructions.md) for the
engineering guide.

Built on **Python 3.14.2** with [`uv`](https://docs.astral.sh/uv/), FastAPI +
Uvicorn, async SQLAlchemy (SQLite by default), and Google Gemini as the LLM.

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

2. Install dependencies:

```bash
uv pip install -r requirements.txt
```

3. Configure the environment. Copy the sample and set your Gemini key:

```bash
cp .env.sample .env
# edit .env and set GEMINI_API_KEY=...
```

`GEMINI_API_KEY` is the only required value; every other setting has a default
(see [`app/core/config.py`](app/core/config.py)).

## Run the app

```bash
uv run uvicorn main:app --reload
```

- **Web UI:** open `http://127.0.0.1:8000/ui/` — ask a question, attach files,
  watch the plan/activity stream, and export a chat as Markdown (**↓ MD** for the
  conversation, **↓ Full** for the full run with plan steps + activity trace).
- **Health check:** `http://127.0.0.1:8000/health`

> `uv run` executes inside the project environment automatically — no manual
> activation needed.

## Configuration

All knobs live in [`.env`](.env.sample) (loaded by `app/core/config.py`). The
ones most worth knowing:

| Setting | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | **Required.** Google Gemini API key. |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Default model (per-session override in the UI). |
| `MAX_AGENT_ITERATIONS` | `120` | Max tool-loop turns for single-agent chat. |
| `MAX_PLAN_ITERATIONS` | `60` | Max plan→execute→evaluate replans per research run. |
| `SUBAGENT_MAX_ITERATIONS` | `60` | Max tool-loop turns per sub-agent. |
| `MAX_PLAN_NODES` | `48` | Hard cap on plan-graph nodes (runaway guard). |
| `SUBAGENT_MAX_REFLECTIONS` | `2` | Self-critique rounds per sub-agent. |

**Unlimited iterations:** the three iteration caps accept `0` (or any
non-positive value) to mean **no limit** — the loop then runs until the model
answers, the run is stopped, the token budget is hit, or the no-progress guard
trips. Use with a token budget configured, since the iteration cap is otherwise
the main runaway guard.

Every model turn (planner, evaluator, reviser, reflection, synthesis, and the
tool loop) is automatically stamped with the current date/time, so the agents
reason about "now", recency, and "latest" against the server clock rather than
the model's training cutoff.

## Test

```bash
uv run pytest -v
```

## Lint & type-check

```bash
uv run ruff check .
uv run mypy --explicit-package-bases .   # app/ is a namespace package
```

## Docker

```bash
docker compose up --build
```

The image uses the `python:3.14.2-slim` base and installs dependencies with `uv`.

## Notes

- For production, replace `--reload` with a production-ready configuration and
  set `ENV=production`, `DEBUG=False`.
- SQLite is the default store; the database URL is configurable (Postgres-capable).
