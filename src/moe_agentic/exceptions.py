"""Custom exceptions for the skill loading framework."""


class SkillError(Exception):
    """Base exception for all skill-related errors."""


class SkillNotFoundError(SkillError):
    """Raised when a skill cannot be found in any search path."""

    def __init__(self, name: str, searched_paths: list[str] | None = None) -> None:
        self.name = name
        self.searched_paths = searched_paths or []
        paths_info = ""
        if self.searched_paths:
            paths_info = f" Searched: {', '.join(self.searched_paths)}"
        super().__init__(f"Skill '{name}' not found.{paths_info}")


class SkillValidationError(SkillError):
    """Raised when a skill fails validation (name, structure, frontmatter)."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"Skill '{name}' validation failed: {reason}")


class SkillParseError(SkillError):
    """Raised when SKILL.md content cannot be parsed (bad YAML, missing fields)."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse '{path}': {reason}")
