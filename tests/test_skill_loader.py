"""Tests for src/moe_agentic/skill_loader.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from moe_agentic.exceptions import (
    SkillNotFoundError,
    SkillParseError,
    SkillValidationError,
)
from moe_agentic.skill_loader import Skill, SkillLoader, SkillMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(
    tmp_path: Path,
    name: str,
    *,
    frontmatter_name: str | None = None,
    description: str = "A test skill.",
    extra_yaml: str = "",
    body: str = "# Hello\n\nWorld.\n",
    subdirs: list[str] | None = None,
    skills_root: str = "skills",
) -> Path:
    """Create a minimal skill directory structure under *tmp_path*."""
    root = tmp_path / skills_root / name
    root.mkdir(parents=True, exist_ok=True)
    fm_name = frontmatter_name or name
    parts = [
        "---",
        f"name: {fm_name}",
        f"description: {description}",
    ]
    if extra_yaml:
        parts.append(extra_yaml)
    parts.append("---")
    parts.append(body)
    content = "\n".join(parts) + "\n"
    (root / "SKILL.md").write_text(content, encoding="utf-8")
    for sub in subdirs or []:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# SkillMetadata / Skill dataclass basics
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_metadata_required_fields(self) -> None:
        meta = SkillMetadata(name="foo", description="bar")
        assert meta.name == "foo"
        assert meta.description == "bar"
        assert meta.license is None
        assert meta.compatibility is None
        assert meta.metadata is None
        assert meta.argument_hint is None

    def test_metadata_all_fields(self) -> None:
        meta = SkillMetadata(
            name="my-skill",
            description="desc",
            license="MIT",
            compatibility=["claude-code", "opencode"],
            metadata={"author": "test"},
            argument_hint="<file>",
        )
        assert meta.license == "MIT"
        assert meta.compatibility == ["claude-code", "opencode"]
        assert meta.metadata == {"author": "test"}
        assert meta.argument_hint == "<file>"

    def test_skill_dataclass(self, tmp_path: Path) -> None:
        meta = SkillMetadata(name="x", description="y")
        skill = Skill(
            metadata=meta,
            content="body",
            path=tmp_path / "SKILL.md",
        )
        assert skill.knowledge_dir is None
        assert skill.tools_dir is None
        assert skill.examples_dir is None


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


class TestValidateName:
    @pytest.mark.parametrize(
        "name",
        [
            "a",
            "abc",
            "my-skill",
            "skill-123",
            "a-b-c-d",
            "a" * 64,
            "0",
            "123",
            "a1b2",
        ],
    )
    def test_valid_names(self, name: str) -> None:
        assert SkillLoader.validate_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "A",
            "MySkill",
            "my_skill",
            "my skill",
            "-leading",
            "trailing-",
            "double--dash",
            "a" * 65,
            "has.dot",
            "has/slash",
            "UPPER",
        ],
    )
    def test_invalid_names(self, name: str) -> None:
        assert SkillLoader.validate_name(name) is False


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_basic(self) -> None:
        content = textwrap.dedent("""\
            ---
            name: my-skill
            description: A great skill.
            ---
            # Body

            Content here.
        """)
        meta, body = SkillLoader.parse_frontmatter(content)
        assert meta.name == "my-skill"
        assert meta.description == "A great skill."
        assert "# Body" in body
        assert "Content here." in body

    def test_multiline_description(self) -> None:
        content = textwrap.dedent("""\
            ---
            name: multi
            description: >
              This is a long
              multiline description.
            ---
            Body.
        """)
        meta, body = SkillLoader.parse_frontmatter(content)
        assert "long" in meta.description
        assert "multiline" in meta.description

    def test_all_optional_fields(self) -> None:
        content = textwrap.dedent("""\
            ---
            name: full
            description: Full skill.
            license: Apache-2.0
            compatibility:
              - claude-code
              - opencode
            metadata:
              author: tester
            argument-hint: "<path>"
            ---
            Body.
        """)
        meta, body = SkillLoader.parse_frontmatter(content)
        assert meta.license == "Apache-2.0"
        assert meta.compatibility == ["claude-code", "opencode"]
        assert meta.metadata == {"author": "tester"}
        assert meta.argument_hint == "<path>"

    def test_compatibility_string_normalised(self) -> None:
        content = textwrap.dedent("""\
            ---
            name: compat
            description: test
            compatibility: opencode
            ---
            Body.
        """)
        meta, _ = SkillLoader.parse_frontmatter(content)
        assert meta.compatibility == ["opencode"]

    def test_missing_opening_fence(self) -> None:
        with pytest.raises(SkillParseError, match="Missing opening"):
            SkillLoader.parse_frontmatter("name: oops\n---\n")

    def test_missing_closing_fence(self) -> None:
        with pytest.raises(SkillParseError, match="Missing closing"):
            SkillLoader.parse_frontmatter("---\nname: oops\n")

    def test_missing_name_field(self) -> None:
        content = "---\ndescription: only desc\n---\n"
        with pytest.raises(SkillParseError, match="missing required field 'name'"):
            SkillLoader.parse_frontmatter(content)

    def test_missing_description_field(self) -> None:
        content = "---\nname: only-name\n---\n"
        with pytest.raises(
            SkillParseError, match="missing required field 'description'"
        ):
            SkillLoader.parse_frontmatter(content)

    def test_malformed_yaml(self) -> None:
        content = "---\n: [broken\n---\n"
        with pytest.raises(SkillParseError, match="Invalid YAML"):
            SkillLoader.parse_frontmatter(content)

    def test_non_mapping_yaml(self) -> None:
        content = "---\n- list-item\n---\n"
        with pytest.raises(SkillParseError, match="must be a YAML mapping"):
            SkillLoader.parse_frontmatter(content)

    def test_bom_handling(self) -> None:
        content = "\ufeff---\nname: bom-skill\ndescription: BOM.\n---\nBody.\n"
        meta, body = SkillLoader.parse_frontmatter(content)
        assert meta.name == "bom-skill"

    def test_argument_hint_underscore_key(self) -> None:
        content = "---\nname: hint\ndescription: test\nargument_hint: '<x>'\n---\n"
        meta, _ = SkillLoader.parse_frontmatter(content)
        assert meta.argument_hint == "<x>"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_discovers_skills(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "alpha")
        _make_skill(tmp_path, "beta")
        loader = SkillLoader(skills_dirs=[tmp_path / "skills"])
        found = loader.discover()
        assert set(found.keys()) == {"alpha", "beta"}

    def test_skips_invalid_dir_names(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "good-name")
        # Create a dir with invalid name
        bad = tmp_path / "skills" / "Bad_Name"
        bad.mkdir(parents=True)
        (bad / "SKILL.md").write_text("---\nname: Bad_Name\ndescription: x\n---\n")
        loader = SkillLoader(skills_dirs=[tmp_path / "skills"])
        found = loader.discover()
        assert "good-name" in found
        assert "Bad_Name" not in found

    def test_first_found_wins(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a" / "alpha"
        dir_b = tmp_path / "b" / "alpha"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        (dir_a / "SKILL.md").write_text("---\nname: alpha\ndescription: first\n---\n")
        (dir_b / "SKILL.md").write_text("---\nname: alpha\ndescription: second\n---\n")
        loader = SkillLoader(skills_dirs=[tmp_path / "a", tmp_path / "b"])
        found = loader.discover()
        assert found["alpha"] == dir_a

    def test_nonexistent_dir_ignored(self, tmp_path: Path) -> None:
        loader = SkillLoader(skills_dirs=[tmp_path / "nonexistent"])
        assert loader.discover() == {}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_skill(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "my-skill", subdirs=["knowledge", "tools", "examples"])
        loader = SkillLoader(skills_dirs=[tmp_path / "skills"])
        skill = loader.load("my-skill")
        assert skill.metadata.name == "my-skill"
        assert skill.metadata.description == "A test skill."
        assert skill.knowledge_dir is not None
        assert skill.tools_dir is not None
        assert skill.examples_dir is not None
        assert "Hello" in skill.content

    def test_load_without_subdirs(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "bare")
        loader = SkillLoader(skills_dirs=[tmp_path / "skills"])
        skill = loader.load("bare")
        assert skill.knowledge_dir is None
        assert skill.tools_dir is None
        assert skill.examples_dir is None

    def test_load_not_found(self, tmp_path: Path) -> None:
        loader = SkillLoader(skills_dirs=[tmp_path / "skills"])
        with pytest.raises(SkillNotFoundError, match="ghost"):
            loader.load("ghost")

    def test_load_invalid_name(self) -> None:
        loader = SkillLoader(skills_dirs=[])
        with pytest.raises(SkillValidationError, match="validation failed"):
            loader.load("BAD_NAME")

    def test_load_name_mismatch(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "dir-name", frontmatter_name="other-name")
        loader = SkillLoader(skills_dirs=[tmp_path / "skills"])
        with pytest.raises(SkillValidationError, match="does not match"):
            loader.load("dir-name")


# ---------------------------------------------------------------------------
# Load all
# ---------------------------------------------------------------------------


class TestLoadAll:
    def test_loads_all_valid(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "one")
        _make_skill(tmp_path, "two")
        loader = SkillLoader(skills_dirs=[tmp_path / "skills"])
        skills = loader.load_all()
        assert set(skills.keys()) == {"one", "two"}

    def test_skips_malformed(self, tmp_path: Path) -> None:
        _make_skill(tmp_path, "good")
        # Create a malformed skill
        bad_dir = tmp_path / "skills" / "bad"
        bad_dir.mkdir(parents=True)
        (bad_dir / "SKILL.md").write_text("no frontmatter here")
        loader = SkillLoader(skills_dirs=[tmp_path / "skills"])
        skills = loader.load_all()
        assert "good" in skills
        assert "bad" not in skills


# ---------------------------------------------------------------------------
# Default search dirs
# ---------------------------------------------------------------------------


class TestDefaultDirs:
    def test_default_dirs_are_built(self) -> None:
        loader = SkillLoader()
        # Should have relative + user dirs
        assert len(loader._search_dirs) > 0

    def test_custom_dirs_override_defaults(self, tmp_path: Path) -> None:
        custom = [tmp_path / "my-skills"]
        loader = SkillLoader(skills_dirs=custom)
        assert len(loader._search_dirs) == 1
        assert loader._search_dirs[0] == custom[0].resolve()


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_not_found_has_searched_paths(self) -> None:
        exc = SkillNotFoundError("x", ["/a", "/b"])
        assert exc.name == "x"
        assert "/a" in str(exc)
        assert "/b" in str(exc)

    def test_validation_has_reason(self) -> None:
        exc = SkillValidationError("x", "bad format")
        assert exc.name == "x"
        assert "bad format" in str(exc)

    def test_parse_has_path(self) -> None:
        exc = SkillParseError("/foo/SKILL.md", "no yaml")
        assert "/foo/SKILL.md" in str(exc)

    def test_all_inherit_from_skill_error(self) -> None:
        from moe_agentic.exceptions import SkillError

        assert issubclass(SkillNotFoundError, SkillError)
        assert issubclass(SkillValidationError, SkillError)
        assert issubclass(SkillParseError, SkillError)
