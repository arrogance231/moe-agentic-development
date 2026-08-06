"""Tests for the SkillLoader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from moe_agentic.loader import SkillLoader, SkillLoadError
from moe_agentic.models import validate_skill_name

_VALID_SKILL_MD = textwrap.dedent("""\
    ---
    name: my-test-skill
    description: A valid test skill.
    argument-hint: <arg>
    ---

    # My Test Skill

    Instructions go here.
""")


def _make_skill(tmp_path: Path, name: str, content: str) -> Path:
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True)
    md_path = skill_dir / "SKILL.md"
    md_path.write_text(content, encoding="utf-8")
    return tmp_path / "skills"


class TestSkillLoader:
    def test_discover(self, tmp_path: Path) -> None:
        skills_dir = _make_skill(tmp_path, "my-test-skill", _VALID_SKILL_MD)
        loader = SkillLoader(skills_dir)
        paths = loader.discover()
        assert len(paths) == 1
        assert paths[0].name == "SKILL.md"

    def test_load_valid(self, tmp_path: Path) -> None:
        skills_dir = _make_skill(tmp_path, "my-test-skill", _VALID_SKILL_MD)
        loader = SkillLoader(skills_dir)
        skill = loader.load_all()[0]
        assert skill.name == "my-test-skill"
        assert skill.metadata.description == "A valid test skill."
        assert "Instructions go here" in skill.body

    def test_load_missing_name(self, tmp_path: Path) -> None:
        bad = "---\ndescription: no name\n---\nbody\n"
        skills_dir = _make_skill(tmp_path, "bad", bad)
        loader = SkillLoader(skills_dir)
        with pytest.raises(SkillLoadError, match="name"):
            loader.load_all()

    def test_load_missing_frontmatter(self, tmp_path: Path) -> None:
        bad = "# No frontmatter\njust markdown\n"
        skills_dir = _make_skill(tmp_path, "bad", bad)
        loader = SkillLoader(skills_dir)
        with pytest.raises(SkillLoadError, match="frontmatter"):
            loader.load_all()

    def test_load_by_name(self, tmp_path: Path) -> None:
        skills_dir = _make_skill(tmp_path, "my-test-skill", _VALID_SKILL_MD)
        loader = SkillLoader(skills_dir)
        skill = loader.load_by_name("my-test-skill")
        assert skill is not None
        assert skill.name == "my-test-skill"

    def test_load_by_name_not_found(self, tmp_path: Path) -> None:
        skills_dir = _make_skill(tmp_path, "my-test-skill", _VALID_SKILL_MD)
        loader = SkillLoader(skills_dir)
        assert loader.load_by_name("nonexistent") is None

    def test_validate_all_clean(self, tmp_path: Path) -> None:
        skills_dir = _make_skill(tmp_path, "my-test-skill", _VALID_SKILL_MD)
        loader = SkillLoader(skills_dir)
        issues = loader.validate_all()
        assert issues == {}

    def test_validate_name_mismatch(self, tmp_path: Path) -> None:
        skills_dir = _make_skill(tmp_path, "wrong-dir", _VALID_SKILL_MD)
        loader = SkillLoader(skills_dir)
        issues = loader.validate_all()
        assert "wrong-dir" in issues
        assert any("does not match" in e for e in issues["wrong-dir"])


class TestValidateSkillName:
    def test_valid_names(self) -> None:
        for name in ("my-skill", "a", "skill-v2", "abc-def-ghi"):
            assert validate_skill_name(name) == [], f"{name} should be valid"

    def test_invalid_names(self) -> None:
        for name in ("My-Skill", "skill_name", "-leading", "trailing-", "a--b", ""):
            assert len(validate_skill_name(name)) > 0, f"{name} should be invalid"

    def test_too_long(self) -> None:
        long_name = "a" * 65
        errors = validate_skill_name(long_name)
        assert any("exceeds" in e for e in errors)
