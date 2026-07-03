"""Python code-execution tool (docs/implementation-plan.md Phase 6).

Runs arbitrary Python in a separate subprocess with a hard timeout and
captured output. Optional pip installs are supported. Each run gets its
own working directory under ``data/pyexec/<session>/`` so files the code
writes are isolated and can be surfaced as artifacts.

Note: this is process-level isolation (separate interpreter + timeout),
not a full security sandbox. For untrusted workloads run the app inside a
container/VM. This is called out as the primary hardening item in the plan.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.logging import get_logger
from app.tools.base import Tool, ToolContext

logger = get_logger(__name__)

DEFAULT_TIMEOUT_S = 30
MAX_TIMEOUT_S = 120
OUTPUT_CHAR_CAP = 8_000
INSTALL_TIMEOUT_S = 180


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= OUTPUT_CHAR_CAP:
        return text, False
    return text[:OUTPUT_CHAR_CAP] + "\n…[truncated]", True


async def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
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
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(
        errors="replace"
    )


async def _install_packages(packages: list[str], cwd: Path) -> dict[str, Any] | None:
    """Best-effort pip install; returns an error dict on failure, else None."""

    cmd = [sys.executable, "-m", "pip", "install", *packages]
    try:
        code, out, err = await _run(cmd, cwd, INSTALL_TIMEOUT_S)
    except TimeoutError:
        return {"error": f"pip install timed out after {INSTALL_TIMEOUT_S}s"}
    if code != 0:
        tail, _ = _truncate(err or out)
        return {"error": f"pip install failed (exit {code})", "pip_output": tail}
    return None


async def _handle(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    code = str(args["code"])
    timeout = min(int(args.get("timeout", DEFAULT_TIMEOUT_S)), MAX_TIMEOUT_S)
    packages = [str(p) for p in args.get("install_packages", []) if str(p).strip()]

    # Resolve to absolute: data_dir may be a relative path, and we run the
    # script with cwd=run_dir, so a relative script path would be re-resolved
    # against the new cwd (doubling the path).
    run_dir = (ctx.data_dir / "pyexec" / str(ctx.session_id) / uuid4().hex).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if packages:
        logger.info("python_exec: installing %s", packages)
        install_error = await _install_packages(packages, run_dir)
        if install_error is not None:
            return install_error

    script_path = run_dir / "script.py"
    script_path.write_text(code, encoding="utf-8")

    logger.info("python_exec: running script in %s", run_dir)
    try:
        exit_code, stdout, stderr = await _run(
            [sys.executable, str(script_path)], run_dir, timeout
        )
    except TimeoutError:
        return {"error": f"Execution timed out after {timeout}s"}

    # Surface any files the script produced (besides the script itself).
    created = [
        p.name for p in run_dir.iterdir() if p.is_file() and p.name != "script.py"
    ]
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


python_exec_tool = Tool(
    name="python_exec",
    description=(
        "Execute Python code in an isolated subprocess and return stdout, "
        "stderr and exit code. Use for calculations, data transformation, "
        "parsing, or plotting. Print results you want to see. Optionally pass "
        "install_packages to pip-install dependencies first. Files the code "
        "writes to the working directory are reported in files_created."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to run."},
            "install_packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional pip packages to install before running.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to run (default 30, max 120).",
            },
        },
        "required": ["code"],
    },
    handler=_handle,
)
