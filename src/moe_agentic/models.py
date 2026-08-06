"""Data models for agentic skills.

This module re-exports the canonical dataclasses from skill_loader and
provides a backward-compatible validate_skill_name helper.
"""

from __future__ import annotations

from moe_agentic.skill_loader import Skill, SkillLoader, SkillMetadata

__all__ = ["Skill", "SkillMetadata", "validate_skill_name"]


def validate_skill_name(name: str) -> list[str]:
    """Return a list of validation errors for a skill name (empty = valid).

    This is a convenience wrapper around :meth:`SkillLoader.validate_name`
    that returns error strings instead of a boolean.

    Args:
        name: Candidate skill name.

    Returns:
        List of validation error strings. Empty list means valid.
    """
    errors: list[str] = []
    if not name:
        errors.append("Skill name is empty.")
        return errors
    if len(name) > 64:
        errors.append(f"Skill name exceeds 64 characters: {len(name)}.")
    if not SkillLoader.validate_name(name):
        if not errors:  # avoid duplicate length message
            errors.append(
                f"Skill name '{name}' must match ^[a-z0-9]+(-[a-z0-9]+)*$ "
                "(lowercase alphanumeric, hyphens between words)."
            )
    return errors
