"""Core skill loading framework.

Discovers, validates, parses, and loads SKILL.md files from multiple
search paths. Supports the universal SKILL.md-with-YAML-frontmatter
format used by Claude Code, OpenCode, and generic agent runtimes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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
        extra: Optional dict of additional/unrecognised frontmatter fields.
    """

    name: str
    description: str
    license: str | None = None
    compatibility: list[str] | None = None
    metadata: dict[str, str] | None = None
    argument_hint: str | None = None
    extra: dict[str, Any] | None = None


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

    # -- convenience properties (used by deploy.py and cli.py) ---------------

    @property
    def name(self) -> str:
        """Shortcut to metadata.name."""
        return self.metadata.name

    @property
    def body(self) -> str:
        """Alias for content (backwards-compatible)."""
        return self.content

    @property
    def source_dir(self) -> Path:
        """Directory containing the SKILL.md file."""
        return self.path.parent

    @property
    def skill_md_path(self) -> Path:
        """Alias for path (backwards-compatible)."""
        return self.path

    @property
    def subdirectories(self) -> list[str]:
        """List of optional subdirectories present (knowledge, tools, examples)."""
        result: list[str] = []
        if self.knowledge_dir is not None:
            result.append("knowledge")
        if self.tools_dir is not None:
            result.append("tools")
        if self.examples_dir is not None:
            result.append("examples")
        return result

    @property
    def has_knowledge(self) -> bool:
        """True if a knowledge/ subdirectory exists."""
        return self.knowledge_dir is not None

    @property
    def has_tools(self) -> bool:
        """True if a tools/ subdirectory exists."""
        return self.tools_dir is not None

    @property
    def has_examples(self) -> bool:
        """True if an examples/ subdirectory exists."""
        return self.examples_dir is not None


# ---------------------------------------------------------------------------
# Name validation utility (module-level, used by CLI)
# ---------------------------------------------------------------------------


def validate_skill_name(name: str) -> list[str]:
    """Return a list of validation errors for a skill name (empty = valid).

    Args:
        name: Candidate skill name.

    Returns:
        List of error strings.  Empty list means the name is valid.
    """
    errors: list[str] = []
    if not name:
        errors.append("Skill name is empty.")
        return errors
    if len(name) > _NAME_MAX_LENGTH:
        errors.append(
            f"Skill name exceeds {_NAME_MAX_LENGTH} characters: {len(name)}."
        )
    if not _NAME_PATTERN.match(name):
        errors.append(
            f"Skill name '{name}' must match ^[a-z0-9]+(-[a-z0-9]+)*$ "
            "(lowercase alphanumeric, hyphens between words)."
        )
    return errors


# ---------------------------------------------------------------------------
# SkillLoader
# ---------------------------------------------------------------------------


class SkillLoader:
    """Discovers, validates, and loads skills from SKILL.md files.

    Supports two modes:

    1. **Single-directory mode** -- pass a single ``Path`` to search one
       skills directory (used by CLI / deployer).
    2. **Multi-directory mode** -- pass a list of ``Path`` via
       *skills_dirs* to search several directories with first-found-wins
       deduplication (used for runtime discovery).

    When neither argument is given, the loader builds a default list from
    well-known relative and user-level locations.
    """

    def __init__(
        self,
        skills_dir: Path | None = None,
        *,
        skills_dirs: list[Path] | None = None,
    ) -> None:
        """Initialise the loader.

        Args:
            skills_dir: Single directory to search (convenience for deploy/CLI).
            skills_dirs: Multiple directories to search (multi-runtime discovery).
                Takes precedence over *skills_dir* if both are given.
        """
        if skills_dirs is not None:
            # Coerce entries defensively so str paths also work.
            self._search_dirs: list[Path] = [
                Path(p).resolve() for p in skills_dirs
            ]
        elif skills_dir is not None:
            self._search_dirs = [skills_dir.resolve()]
        else:
            self._search_dirs = self._build_default_search_dirs()

        # Expose the primary directory for callers that need it.
        self.skills_dir: Path = (
            self._search_dirs[0] if self._search_dirs else Path.cwd()
        )

    # -- public API ---------------------------------------------------------

    def discover(self) -> dict[str, Path] | list[Path]:
        """Discover all available skills by scanning search paths.

        Returns:
            A mapping of validated skill name to the directory containing
            its ``SKILL.md``.  First-found-wins when the same name appears
            in multiple search paths.
        """
        return {
            name: path
            for name, path in self._scan_all().items()
            if self._is_valid_name(name)
        }

    def discover_all(self) -> dict[str, Path]:
        """Discover every skill directory, including invalid names.

        Returns:
            A mapping of directory name to directory path for every
            directory containing a ``SKILL.md``, regardless of whether
            the name conforms to the skill-name spec.  First-found-wins
            when the same name appears in multiple search paths.
        """
        return self._scan_all()

    def discover_paths(self) -> list[Path]:
        """Return sorted list of SKILL.md paths found."""
        found = self.discover()
        if isinstance(found, dict):
            return [d / SKILL_FILENAME for d in found.values()]
        return found  # pragma: no cover

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
        if not self._is_valid_name(name):
            raise SkillValidationError(
                name,
                f"Name must match /{_NAME_PATTERN.pattern}/ and be "
                f"<= {_NAME_MAX_LENGTH} chars.",
            )

        discovered = self.discover()
        if isinstance(discovered, dict) and name not in discovered:
            raise SkillNotFoundError(
                name,
                searched_paths=[str(p) for p in self._search_dirs],
            )

        skill_dir = discovered[name] if isinstance(discovered, dict) else None
        if skill_dir is None:
            raise SkillNotFoundError(
                name,
                searched_paths=[str(p) for p in self._search_dirs],
            )
        return self._load_from_dir(skill_dir, expected_name=name)

    def load_by_name(self, name: str) -> Skill | None:
        """Load a skill by name.  Returns None if not found.

        Args:
            name: The skill name to look up.

        Returns:
            A :class:`Skill` or *None*.
        """
        for search_dir in self._search_dirs:
            candidate = search_dir / name / SKILL_FILENAME
            if candidate.is_file():
                try:
                    return self._load_from_dir(
                        candidate.parent, expected_name=name
                    )
                except (SkillParseError, SkillValidationError):
                    return None
        return None

    def load_all(self) -> dict[str, Skill]:
        """Load every discovered skill, keyed by skill name.

        Returns:
            A dict mapping skill name to :class:`Skill`.  Skills that
            fail to parse are silently skipped.
        """
        skills: dict[str, Skill] = {}
        discovered = self.discover()
        items = discovered.items() if isinstance(discovered, dict) else []
        for name, skill_dir in items:
            try:
                skills[name] = self._load_from_dir(
                    skill_dir, expected_name=name
                )
            except (SkillParseError, SkillValidationError):
                continue
        return skills

    def load_all_dict(self) -> dict[str, Skill]:
        """Load every discovered skill as a name-keyed dict.

        Now redundant since :meth:`load_all` already returns a name-keyed
        dict; kept for backwards compatibility.

        Returns:
            A mapping of skill name to :class:`Skill`.
        """
        return self.load_all()

    def iter_skills(self) -> Iterator[Skill]:
        """Discover and lazily yield skills."""
        discovered = self.discover()
        items = discovered.items() if isinstance(discovered, dict) else []
        for name, skill_dir in items:
            try:
                yield self._load_from_dir(skill_dir, expected_name=name)
            except (SkillParseError, SkillValidationError):
                continue

    def validate_all(self) -> dict[str, list[str]]:
        """Validate all skill directories, including invalid names.

        Unlike :meth:`discover`, this walks the raw scan so directories
        with non-conforming names are checked too.

        Returns:
            Dict mapping skill directory name to list of validation errors.
            Directories with no errors are omitted.
        """
        issues: dict[str, list[str]] = {}
        for dir_name, skill_dir in self._scan_all().items():
            errors: list[str] = []

            # The directory name itself must conform to the naming spec.
            errors.extend(validate_skill_name(dir_name))

            try:
                skill = self._load_from_dir(skill_dir, expected_name=dir_name)
            except SkillParseError as exc:
                errors.append(str(exc))
                issues[dir_name] = errors
                continue
            except SkillValidationError as exc:
                errors.append(str(exc))
                issues[dir_name] = errors
                continue
            if skill.name != dir_name:
                errors.append(
                    f"Skill name '{skill.name}' does not match "
                    f"directory name '{dir_name}'."
                )
            if not skill.content.strip():
                errors.append(
                    "SKILL.md body is empty (no instructions after frontmatter)."
                )

            if errors:
                issues[dir_name] = errors
        return issues

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
        return SkillLoader._is_valid_name(name)

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
            raise SkillParseError(
                "<content>", "Missing opening '---' frontmatter fence."
            )

        # Find closing fence
        closing_idx: int | None = None
        for idx in range(1, len(lines)):
            if lines[idx].rstrip() == _FRONTMATTER_FENCE:
                closing_idx = idx
                break

        if closing_idx is None:
            raise SkillParseError(
                "<content>", "Missing closing '---' frontmatter fence."
            )

        yaml_block = "\n".join(lines[1:closing_idx])
        body = "\n".join(lines[closing_idx + 1 :])

        # Parse YAML
        try:
            raw: Any = yaml.safe_load(yaml_block)
        except yaml.YAMLError as exc:
            raise SkillParseError(
                "<content>", f"Invalid YAML in frontmatter: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise SkillParseError(
                "<content>", "Frontmatter must be a YAML mapping."
            )

        # Required fields
        if "name" not in raw:
            raise SkillParseError(
                "<content>",
                "Frontmatter missing required field 'name'.",
            )
        if "description" not in raw:
            raise SkillParseError(
                "<content>",
                "Frontmatter missing required field 'description'.",
            )

        # Normalise compatibility to list
        compat = raw.pop("compatibility", None)
        if isinstance(compat, str):
            compat = [compat]

        # Normalise argument-hint (YAML key uses hyphen, Python underscore)
        argument_hint = raw.pop("argument-hint", None) or raw.pop(
            "argument_hint", None
        )

        # Extract known fields, remainder goes to extra
        name_val = str(raw.pop("name"))
        desc_val = str(raw.pop("description"))
        license_val = raw.pop("license", None)
        metadata_val = raw.pop("metadata", None)

        # Anything left in raw is extra
        extra = raw if raw else None

        meta = SkillMetadata(
            name=name_val,
            description=desc_val,
            license=license_val,
            compatibility=compat,
            metadata=metadata_val,
            argument_hint=argument_hint,
            extra=extra,
        )
        return meta, body.lstrip("\n")

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _is_valid_name(name: str) -> bool:
        """Check if a name is valid per the OpenCode spec."""
        if not name or len(name) > _NAME_MAX_LENGTH:
            return False
        return _NAME_PATTERN.match(name) is not None

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

    def _scan_all(self) -> dict[str, Path]:
        """Scan search paths for ALL skill directories, valid or not.

        Returns:
            A mapping of directory name to directory path for every
            directory containing a ``SKILL.md``, regardless of name
            validity.  First-found-wins when the same name appears in
            multiple search paths.
        """
        found: dict[str, Path] = {}
        for search_dir in self._search_dirs:
            if not search_dir.is_dir():
                continue
            for child in sorted(search_dir.iterdir()):
                skill_file = child / SKILL_FILENAME
                if child.is_dir() and skill_file.is_file():
                    if child.name not in found:
                        found[child.name] = child
        return found

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
        if not self._is_valid_name(meta.name):
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
            knowledge_dir=(
                knowledge_dir.resolve() if knowledge_dir.is_dir() else None
            ),
            tools_dir=tools_dir.resolve() if tools_dir.is_dir() else None,
            examples_dir=(
                examples_dir.resolve() if examples_dir.is_dir() else None
            ),
        )
