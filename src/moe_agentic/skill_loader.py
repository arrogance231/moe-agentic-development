"""Core skill loading framework.

Discovers, validates, parses, and loads SKILL.md files from multiple
search paths. Supports the universal SKILL.md-with-YAML-frontmatter
format used by Claude Code, OpenCode, and generic agent runtimes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from moe_agentic.exceptions import (
    SkillNotFoundError,
    SkillParseError,
    SkillValidationError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_NAME_MAX_LENGTH: int = 64
_FRONTMATTER_FENCE: str = "---"

# Default relative search directories (resolved against cwd)
_DEFAULT_RELATIVE_DIRS: list[str] = [
    "skills",
    ".opencode/skills",
    ".claude/skills",
    ".agents/skills",
]

# Default user-level search directories (resolved against ~)
_DEFAULT_USER_DIRS: list[str] = [
    ".config/opencode/skills",
    ".claude/skills",
    ".agents/skills",
]

SKILL_FILENAME: str = "SKILL.md"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillMetadata:
    """Parsed YAML frontmatter from a SKILL.md file.

    Attributes:
        name: Kebab-case skill identifier (validated against OpenCode spec).
        description: Human-readable description of the skill's purpose.
        license: Optional SPDX license identifier.
        compatibility: Optional list of runtime identifiers the skill supports.
        metadata: Optional dict of arbitrary key-value metadata.
        argument_hint: Optional hint shown when the skill is invoked.
    """

    name: str
    description: str
    license: str | None = None
    compatibility: list[str] | None = None
    metadata: dict[str, str] | None = None
    argument_hint: str | None = None


@dataclass(frozen=True)
class Skill:
    """A fully loaded skill with its metadata, content, and resource paths.

    Attributes:
        metadata: Parsed frontmatter metadata.
        content: Full markdown body after the frontmatter block.
        path: Absolute path to the SKILL.md file.
        knowledge_dir: Path to the ``knowledge/`` subdirectory, if it exists.
        tools_dir: Path to the ``tools/`` subdirectory, if it exists.
        examples_dir: Path to the ``examples/`` subdirectory, if it exists.
    """

    metadata: SkillMetadata
    content: str
    path: Path
    knowledge_dir: Path | None = None
    tools_dir: Path | None = None
    examples_dir: Path | None = None


# ---------------------------------------------------------------------------
# SkillLoader
# ---------------------------------------------------------------------------


class SkillLoader:
    """Discovers, validates, and loads skills from SKILL.md files.

    Search order:
        1. Explicit *skills_dirs* passed to the constructor.
        2. Default relative directories under the current working directory.
        3. Default user-level directories under ``Path.home()``.

    Duplicate skill names are resolved by first-found-wins across the
    ordered search paths.
    """

    def __init__(self, skills_dirs: list[Path] | None = None) -> None:
        """Initialise the loader with an ordered list of search paths.

        Args:
            skills_dirs: Explicit directories to search. When *None*,
                the loader builds a default list from well-known relative
                and user-level locations.
        """
        if skills_dirs is not None:
            self._search_dirs: list[Path] = [p.resolve() for p in skills_dirs]
        else:
            self._search_dirs = self._build_default_search_dirs()

    # -- public API ---------------------------------------------------------

    def discover(self) -> dict[str, Path]:
        """Discover all available skills by scanning search paths.

        Returns:
            A mapping of validated skill name to the directory containing
            its ``SKILL.md``.  First-found-wins when the same name appears
            in multiple search paths.
        """
        found: dict[str, Path] = {}
        for search_dir in self._search_dirs:
            if not search_dir.is_dir():
                continue
            for child in sorted(search_dir.iterdir()):
                skill_file = child / SKILL_FILENAME
                if child.is_dir() and skill_file.is_file():
                    # Use directory name as the candidate skill name
                    dir_name = child.name
                    if self.validate_name(dir_name) and dir_name not in found:
                        found[dir_name] = child
        return found

    def load(self, name: str) -> Skill:
        """Load a single skill by name.

        Args:
            name: The kebab-case skill name to load.

        Returns:
            A fully populated :class:`Skill` instance.

        Raises:
            SkillValidationError: If *name* does not pass validation.
            SkillNotFoundError: If no skill with that name can be found.
            SkillParseError: If the ``SKILL.md`` content cannot be parsed.
        """
        if not self.validate_name(name):
            raise SkillValidationError(
                name,
                f"Name must match /{_NAME_PATTERN.pattern}/ and be "
                f"<= {_NAME_MAX_LENGTH} chars.",
            )

        discovered = self.discover()
        if name not in discovered:
            raise SkillNotFoundError(
                name,
                searched_paths=[str(p) for p in self._search_dirs],
            )

        skill_dir = discovered[name]
        return self._load_from_dir(skill_dir, expected_name=name)

    def load_all(self) -> dict[str, Skill]:
        """Load every discovered skill.

        Returns:
            A mapping of skill name to :class:`Skill`.  Skills that fail
            to parse are silently skipped (callers who need strict loading
            should use :meth:`load` individually).
        """
        skills: dict[str, Skill] = {}
        for name, skill_dir in self.discover().items():
            try:
                skills[name] = self._load_from_dir(skill_dir, expected_name=name)
            except (SkillParseError, SkillValidationError):
                # Skip malformed skills in bulk loading
                continue
        return skills

    @staticmethod
    def validate_name(name: str) -> bool:
        """Check that *name* conforms to the OpenCode skill-name spec.

        Rules:
            - Matches ``^[a-z0-9]+(-[a-z0-9]+)*$``.
            - At most 64 characters long.

        Args:
            name: Candidate skill name.

        Returns:
            *True* if valid, *False* otherwise.
        """
        if not name or len(name) > _NAME_MAX_LENGTH:
            return False
        return _NAME_PATTERN.match(name) is not None

    @staticmethod
    def parse_frontmatter(content: str) -> tuple[SkillMetadata, str]:
        """Parse YAML frontmatter from the raw text of a SKILL.md file.

        The frontmatter block must be delimited by ``---`` on its own line
        at the very start of the file.

        Args:
            content: Full text content of the SKILL.md file.

        Returns:
            A 2-tuple of (:class:`SkillMetadata`, body markdown string).

        Raises:
            SkillParseError: If frontmatter is missing or YAML is malformed.
        """
        stripped = content.lstrip("\ufeff")  # strip BOM if present
        lines = stripped.split("\n")

        # First non-empty line must be the opening fence
        if not lines or lines[0].rstrip() != _FRONTMATTER_FENCE:
            raise SkillParseError("<content>", "Missing opening '---' frontmatter fence.")

        # Find closing fence
        closing_idx: int | None = None
        for idx in range(1, len(lines)):
            if lines[idx].rstrip() == _FRONTMATTER_FENCE:
                closing_idx = idx
                break

        if closing_idx is None:
            raise SkillParseError("<content>", "Missing closing '---' frontmatter fence.")

        yaml_block = "\n".join(lines[1:closing_idx])
        body = "\n".join(lines[closing_idx + 1 :])

        # Parse YAML
        try:
            raw: Any = yaml.safe_load(yaml_block)
        except yaml.YAMLError as exc:
            raise SkillParseError("<content>", f"Invalid YAML in frontmatter: {exc}") from exc

        if not isinstance(raw, dict):
            raise SkillParseError("<content>", "Frontmatter must be a YAML mapping.")

        # Required fields
        if "name" not in raw:
            raise SkillParseError("<content>", "Frontmatter missing required field 'name'.")
        if "description" not in raw:
            raise SkillParseError("<content>", "Frontmatter missing required field 'description'.")

        # Normalise compatibility to list
        compat = raw.get("compatibility")
        if isinstance(compat, str):
            compat = [compat]

        # Normalise argument-hint (YAML key uses hyphen, Python uses underscore)
        argument_hint = raw.get("argument-hint") or raw.get("argument_hint")

        meta = SkillMetadata(
            name=str(raw["name"]),
            description=str(raw["description"]),
            license=raw.get("license"),
            compatibility=compat,
            metadata=raw.get("metadata"),
            argument_hint=argument_hint,
        )
        return meta, body.lstrip("\n")

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _build_default_search_dirs() -> list[Path]:
        """Build the default ordered list of search directories."""
        cwd = Path.cwd()
        home = Path.home()
        dirs: list[Path] = []

        for rel in _DEFAULT_RELATIVE_DIRS:
            dirs.append((cwd / rel).resolve())

        for rel in _DEFAULT_USER_DIRS:
            dirs.append((home / rel).resolve())

        return dirs

    def _load_from_dir(self, skill_dir: Path, expected_name: str) -> Skill:
        """Load a :class:`Skill` from its directory.

        Args:
            skill_dir: Directory containing ``SKILL.md``.
            expected_name: The name we expect to find in the frontmatter.

        Returns:
            A populated :class:`Skill`.

        Raises:
            SkillParseError: If the file cannot be read or parsed.
            SkillValidationError: If the frontmatter name doesn't match
                the directory name.
        """
        skill_file = skill_dir / SKILL_FILENAME
        try:
            raw_content = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise SkillParseError(
                str(skill_file), f"Cannot read file: {exc}"
            ) from exc

        meta, body = self.parse_frontmatter(raw_content)

        # Validate: directory name must match frontmatter name
        if meta.name != expected_name:
            raise SkillValidationError(
                expected_name,
                f"Directory name '{expected_name}' does not match "
                f"frontmatter name '{meta.name}'.",
            )

        # Validate the frontmatter name itself
        if not self.validate_name(meta.name):
            raise SkillValidationError(
                meta.name,
                f"Frontmatter name '{meta.name}' does not conform to "
                f"the naming spec (/{_NAME_PATTERN.pattern}/, "
                f"<= {_NAME_MAX_LENGTH} chars).",
            )

        # Detect optional subdirectories
        knowledge_dir = skill_dir / "knowledge"
        tools_dir = skill_dir / "tools"
        examples_dir = skill_dir / "examples"

        return Skill(
            metadata=meta,
            content=body,
            path=skill_file.resolve(),
            knowledge_dir=knowledge_dir.resolve() if knowledge_dir.is_dir() else None,
            tools_dir=tools_dir.resolve() if tools_dir.is_dir() else None,
            examples_dir=examples_dir.resolve() if examples_dir.is_dir() else None,
        )
