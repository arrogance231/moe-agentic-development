"""Shared utility functions for the MoE Agentic Development framework."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def read_text_safe(path: Path, encoding: str = "utf-8") -> str:
    """Read a text file, stripping BOM if present.

    Args:
        path: File to read.
        encoding: Text encoding (default utf-8).

    Returns:
        File content with BOM stripped.
    """
    text = path.read_text(encoding=encoding)
    return text.lstrip("\ufeff")
