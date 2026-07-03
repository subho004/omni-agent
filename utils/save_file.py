from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


def _ensure_dir(dir_path: Optional[Union[str, Path]]) -> Path:
    p = Path(dir_path) if dir_path else Path.cwd()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(
    content: Any,
    filename: str,
    dir_path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> Path:
    """Save any Python object as JSON.

    - `content`: anything serializable by `json.dump` (or a Python object; `json` will attempt to serialize it).
    - `filename`: name without extension (if extension provided it will be stripped).
    - `dir_path`: optional directory to save into; defaults to current working directory.

    Returns the Path to the written file. Raises on failure.
    """
    stem = Path(filename).stem
    out_dir = _ensure_dir(dir_path)
    out_path = out_dir / f"{stem}.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=indent)
        return out_path
    except Exception:
        logger.exception("Failed to save JSON to %s", out_path)
        raise


def save_text(
    content: Any, filename: str, dir_path: Optional[Union[str, Path]] = None
) -> Path:
    """Save content as UTF-8 text.

    - `content`: string or any object; non-strings are converted with `str()`.
    - `filename`: name without extension (if extension provided it will be stripped).
    - `dir_path`: optional directory to save into; defaults to current working directory.

    Returns the Path to the written file. Raises on failure.
    """
    stem = Path(filename).stem
    out_dir = _ensure_dir(dir_path)
    out_path = out_dir / f"{stem}.txt"
    try:
        text = content if isinstance(content, str) else str(content)
        out_path.write_text(text, encoding="utf-8")
        return out_path
    except Exception:
        logger.exception("Failed to save text to %s", out_path)
        raise
