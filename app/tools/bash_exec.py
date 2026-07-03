"""Bash command-execution tool.

Runs a shell command in a separate subprocess with a hard timeout and
captured output. Each run gets its own working directory under
``data/bashexec/<session>/`` so files the command writes are isolated and
can be surfaced. Mirrors ``python_exec`` for calculations/data work that is
easier to express as shell (curl, jq, grep, file wrangling, git, etc.).

Note: this is process-level isolation (separate shell + timeout), not a
security sandbox. The command runs with the app's own privileges — for
untrusted workloads run the app inside a container/VM.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

DEFAULT_TIMEOUT_S = 30
MAX_TIMEOUT_S = 120
OUTPUT_CHAR_CAP = 8_000


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= OUTPUT_CHAR_CAP:
        return text, False
    return text[:OUTPUT_CHAR_CAP] + "\n…[truncated]", True


async def _run(command: str, cwd: Path, timeout: int) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    command = str(args["command"]).strip()
    if not command:
        return {"error": "No command provided."}
    timeout = min(int(args.get("timeout", DEFAULT_TIMEOUT_S)), MAX_TIMEOUT_S)

    # Absolute working dir: data_dir may be relative and we run with cwd set,
    # so a relative path would be re-resolved against the new cwd.
    run_dir = (ctx.data_dir / "bashexec" / str(ctx.session_id) / uuid4().hex).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("bash_exec: running in %s", run_dir)
    try:
        exit_code, stdout, stderr = await _run(command, run_dir, timeout)
    except TimeoutError:
        return {"error": f"Command timed out after {timeout}s"}

    created = [p.name for p in run_dir.iterdir() if p.is_file()]
    stdout_text, stdout_trunc = _truncate(stdout)
    stderr_text, stderr_trunc = _truncate(stderr)

    return {
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "truncated": stdout_trunc or stderr_trunc,
        "files_created": created,
        "workdir": str(run_dir),
    }


bash_exec_tool = Tool(
    name="bash_exec",
    description=(
        "Execute a shell (bash) command in an isolated subprocess and return "
        "stdout, stderr and exit code. Use for quick data wrangling with "
        "standard CLI tools (curl, jq, grep, sed, awk, sort, file, git). Runs "
        "in a fresh per-run working directory; files the command writes are "
        "reported in files_created. For multi-step logic or heavy computation "
        "prefer python_exec."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run (bash syntax).",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to run (default 30, max 120).",
            },
        },
        "required": ["command"],
    },
    handler=_handle,
    timeout=float(MAX_TIMEOUT_S),
)
