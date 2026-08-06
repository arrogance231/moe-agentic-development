"""Skill loader - discovers and parses SKILL.md files from a skills directory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import yaml

from moe_agentic.models import Skill, SkillMetadata, validate_skill_name

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillLoadError(Exception):
    """Raised when a SKILL.md file cannot be parsed."""


def _parse_skill_md(path: Path) -> tuple[SkillMetadata, str]:
    """Parse a SKILL.md file into metadata and body text.

    Args:
        path: Path to the SKILL.md file.

    Returns:
        A tuple of (SkillMetadata, body_markdown).

    Raises:
        SkillLoadError: If the file cannot be parsed or is missing required fields.
    """
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise SkillLoadError(f"{path}: missing YAML frontmatter (---).")

    try:
        raw: dict = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillLoadError(f"{path}: invalid YAML frontmatter: {exc}") from exc

    name = raw.pop("name", None)
    if not name:
        raise SkillLoadError(f"{path}: frontmatter missing required 'name' field.")

    description = raw.pop("description", None)
    if not description:
        raise SkillLoadError(f"{path}: frontmatter missing required 'description' field.")

    meta = SkillMetadata(
        name=str(name),
        description=str(description),
        argument_hint=str(raw.pop("argument-hint", raw.pop("argument_hint", ""))),
        license=str(raw.pop("license", "")),
        compatibility=raw.pop("compatibility", []),
        extra=raw,
    )
    body = text[match.end():]
    return meta, body


class SkillLoader:
    """Discovers and loads skills from a directory tree.

    Expected layout::

        skills_dir/
          skill-name/
            SKILL.md
            knowledge/   (optional)
            tools/       (optional)
            examples/    (optional)
    """

    def __init__(self, skills_dir: Path) -> None:
        """Initialize the loader.

        Args:
            skills_dir: Root directory containing skill subdirectories.
        """
        self.skills_dir = skills_dir

    def discover(self) -> list[Path]:
        """Return sorted list of SKILL.md paths found under skills_dir."""
        if not self.skills_dir.is_dir():
            return []
        return sorted(self.skills_dir.glob("*/SKILL.md"))

    def load(self, skill_md_path: Path) -> Skill:
        """Load a single skill from its SKILL.md path.

        Args:
            skill_md_path: Path to the SKILL.md file.

        Returns:
            A fully populated Skill object.

        Raises:
            SkillLoadError: If the file cannot be parsed.
        """
        meta, body = _parse_skill_md(skill_md_path)
        return Skill(
            metadata=meta,
            body=body,
            source_dir=skill_md_path.parent,
            skill_md_path=skill_md_path,
        )

    def load_all(self) -> list[Skill]:
        """Discover and load all skills. Raises on first error."""
        return [self.load(p) for p in self.discover()]

    def iter_skills(self) -> Iterator[Skill]:
        """Discover and lazily yield skills."""
        for p in self.discover():
            yield self.load(p)

    def load_by_name(self, name: str) -> Skill | None:
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
                    f"Skill name '{skill.name}' does not match directory name '{dir_name}'."
                )
            if not skill.body.strip():
                errors.append("SKILL.md body is empty (no instructions after frontmatter).")

            if errors:
                issues[dir_name] = errors
        return issues
