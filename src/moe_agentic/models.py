"""Data models for agentic skills."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Skill name validation: lowercase alphanumeric with hyphens, max 64 chars.
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NAME_MAX_LEN = 64


@dataclass
class SkillMetadata:
    """Parsed YAML frontmatter of a SKILL.md file."""

    name: str
    description: str
    argument_hint: str = ""
    license: str = ""
    compatibility: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """A loaded skill with its metadata, body, and filesystem location."""

    metadata: SkillMetadata
    body: str
    source_dir: Path
    skill_md_path: Path

    @property
    def name(self) -> str:
        """Shortcut to metadata.name."""
        return self.metadata.name

    @property
    def has_knowledge(self) -> bool:
        return (self.source_dir / "knowledge").is_dir()

    @property
    def has_tools(self) -> bool:
        return (self.source_dir / "tools").is_dir()

    @property
    def has_examples(self) -> bool:
        return (self.source_dir / "examples").is_dir()

    @property
    def subdirectories(self) -> list[str]:
        """List of optional subdirectories present (knowledge, tools, examples)."""
        return [
            d
            for d in ("knowledge", "tools", "examples")
            if (self.source_dir / d).is_dir()
        ]


def validate_skill_name(name: str) -> list[str]:
    """Return a list of validation errors for a skill name (empty = valid)."""
    errors: list[str] = []
    if not name:
        errors.append("Skill name is empty.")
        return errors
    if len(name) > _NAME_MAX_LEN:
        errors.append(f"Skill name exceeds {_NAME_MAX_LEN} characters: {len(name)}.")
    if not _NAME_RE.match(name):
        errors.append(
            f"Skill name '{name}' must match ^[a-z0-9]+(-[a-z0-9]+)*$ "
            "(lowercase alphanumeric, hyphens between words)."
        )
    return errors
