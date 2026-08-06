"""Skill loader adapter - wraps SkillLoader for backward-compatible API.

The canonical implementation lives in :mod:`moe_agentic.skill_loader`.
This module exposes a single-directory SkillLoader interface and
a SkillLoadError for use by deploy.py and cli.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from moe_agentic.exceptions import SkillParseError
from moe_agentic.skill_loader import (
    Skill as _CanonicalSkill,
)
from moe_agentic.skill_loader import (
    SkillLoader as _CanonicalLoader,
)
from moe_agentic.skill_loader import (
    SkillMetadata as _CanonicalMeta,
)

__all__ = ["SkillLoadError", "SkillLoader"]


class SkillLoadError(Exception):
    """Raised when a SKILL.md file cannot be parsed."""


@dataclass
class _CompatSkill:
    """Backward-compatible Skill with .name, .body, .source_dir, .subdirectories."""

    metadata: _CanonicalMeta
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


def _to_compat(skill: _CanonicalSkill) -> _CompatSkill:
    """Convert a canonical Skill to the backward-compatible format."""
    return _CompatSkill(
        metadata=skill.metadata,
        body=skill.content,
        source_dir=skill.path.parent,
        skill_md_path=skill.path,
    )


class SkillLoader:
    """Single-directory skill loader (backward-compatible API).

    Wraps the canonical multi-path :class:`~moe_agentic.skill_loader.SkillLoader`
    but exposes a simpler single-directory interface used by deploy.py and cli.py.
    """

    def __init__(self, skills_dir: Path) -> None:
        """Initialize the loader.

        Args:
            skills_dir: Root directory containing skill subdirectories.
        """
        self.skills_dir = skills_dir
        self._canonical = _CanonicalLoader(skills_dirs=[skills_dir])

    def discover(self) -> list[Path]:
        """Return sorted list of SKILL.md paths found under skills_dir."""
        found = self._canonical.discover()
        return sorted(d / "SKILL.md" for d in found.values())

    def load(self, skill_md_path: Path) -> _CompatSkill:
        """Load a single skill from its SKILL.md path.

        Args:
            skill_md_path: Path to the SKILL.md file.

        Returns:
            A backward-compatible Skill object.

        Raises:
            SkillLoadError: If the file cannot be parsed.
        """
        try:
            raw_content = skill_md_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SkillLoadError(f"{skill_md_path}: cannot read file: {exc}") from exc

        try:
            meta, body = _CanonicalLoader.parse_frontmatter(raw_content)
        except SkillParseError as exc:
            raise SkillLoadError(f"{skill_md_path}: {exc.reason}") from exc

        return _CompatSkill(
            metadata=meta,
            body=body,
            source_dir=skill_md_path.parent,
            skill_md_path=skill_md_path,
        )

    def load_all(self) -> list[_CompatSkill]:
        """Discover and load all skills. Raises on first error."""
        return [self.load(p) for p in self.discover()]

    def iter_skills(self) -> Iterator[_CompatSkill]:
        """Discover and lazily yield skills."""
        for p in self.discover():
            yield self.load(p)

    def load_by_name(self, name: str) -> _CompatSkill | None:
        """Load a skill by directory name. Returns None if not found."""
        candidate = self.skills_dir / name / "SKILL.md"
        if candidate.is_file():
            return self.load(candidate)
        return None

    def validate_all(self) -> dict[str, list[str]]:
        """Validate all discovered skills.

        Returns:
            Dict mapping skill directory name to list of validation errors.
            Skills with no errors are omitted.
        """
        from moe_agentic.models import validate_skill_name

        issues: dict[str, list[str]] = {}
        for md_path in self.discover():
            dir_name = md_path.parent.name
            errors: list[str] = []
            try:
                skill = self.load(md_path)
            except SkillLoadError as exc:
                errors.append(str(exc))
                issues[dir_name] = errors
                continue

            errors.extend(validate_skill_name(skill.name))
            if skill.name != dir_name:
                errors.append(
                    f"Skill name '{skill.name}' does not match directory name "
                    f"'{dir_name}'."
                )
            if not skill.body.strip():
                errors.append(
                    "SKILL.md body is empty (no instructions after frontmatter)."
                )

            if errors:
                issues[dir_name] = errors
        return issues
