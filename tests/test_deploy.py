"""Tests for SkillDeployer."""

from __future__ import annotations

import textwrap
from pathlib import Path

from rich.console import Console

from moe_agentic.deploy import (
    DeployTarget,
    SkillDeployer,
    resolve_skills_dir,
)

# -- Fixtures ---------------------------------------------------------------

_SKILL_MD = textwrap.dedent("""\
    ---
    name: test-skill
    description: A test skill for unit testing.
    argument-hint: <test-arg>
    ---

    # Test Skill

    You are a test skill. Do test things.
""")

_SKILL2_MD = textwrap.dedent("""\
    ---
    name: other-skill
    description: Another test skill.
    ---

    # Other Skill

    Do other things.
""")


def _make_skill_tree(tmp_path: Path) -> Path:
    """Create a minimal skills directory with two skills."""
    skills_dir = tmp_path / "skills"

    # Skill 1: has knowledge and tools
    s1 = skills_dir / "test-skill"
    s1.mkdir(parents=True)
    (s1 / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    (s1 / "knowledge").mkdir()
    (s1 / "knowledge" / "data.md").write_text("# Knowledge", encoding="utf-8")
    (s1 / "tools").mkdir()
    (s1 / "tools" / "helper.py").write_text("print('hi')", encoding="utf-8")

    # Skill 2: minimal, no subdirs
    s2 = skills_dir / "other-skill"
    s2.mkdir(parents=True)
    (s2 / "SKILL.md").write_text(_SKILL2_MD, encoding="utf-8")

    return skills_dir


def _quiet_console() -> Console:
    return Console(quiet=True)


# -- resolve_skills_dir tests -----------------------------------------------


class TestResolveSkillsDir:
    def test_project_local_claude(self, tmp_path: Path) -> None:
        result = resolve_skills_dir(
            DeployTarget.CLAUDE, global_install=False, project_root=tmp_path
        )
        assert result == tmp_path / ".claude" / "skills"

    def test_project_local_opencode(self, tmp_path: Path) -> None:
        result = resolve_skills_dir(
            DeployTarget.OPENCODE, global_install=False, project_root=tmp_path
        )
        assert result == tmp_path / ".opencode" / "skills"

    def test_project_local_agents(self, tmp_path: Path) -> None:
        result = resolve_skills_dir(
            DeployTarget.AGENTS, global_install=False, project_root=tmp_path
        )
        assert result == tmp_path / ".agents" / "skills"

    def test_global_claude(self) -> None:
        result = resolve_skills_dir(DeployTarget.CLAUDE, global_install=True)
        assert result == Path.home() / ".claude" / "skills"

    def test_global_opencode(self) -> None:
        result = resolve_skills_dir(DeployTarget.OPENCODE, global_install=True)
        assert result == Path.home() / ".config" / "opencode" / "skills"


# -- SkillDeployer tests ----------------------------------------------------


class TestSkillDeployer:
    def test_deploy_single_target(self, tmp_path: Path) -> None:
        skills_dir = _make_skill_tree(tmp_path)
        deployer = SkillDeployer(
            skills_dir=skills_dir,
            project_root=tmp_path,
            console=_quiet_console(),
        )
        result = deployer.deploy(
            targets=[DeployTarget.CLAUDE],
            force=True,
        )
        assert result.success
        assert "test-skill" in result.skills_deployed
        assert "other-skill" in result.skills_deployed

        # Check files exist
        dest = tmp_path / ".claude" / "skills" / "test-skill"
        assert (dest / "SKILL.md").is_file()
        assert (dest / "knowledge" / "data.md").is_file()
        assert (dest / "tools" / "helper.py").is_file()

        # Skill 2 has no subdirs
        dest2 = tmp_path / ".claude" / "skills" / "other-skill"
        assert (dest2 / "SKILL.md").is_file()
        assert not (dest2 / "knowledge").exists()

    def test_deploy_all_targets(self, tmp_path: Path) -> None:
        skills_dir = _make_skill_tree(tmp_path)
        deployer = SkillDeployer(
            skills_dir=skills_dir,
            project_root=tmp_path,
            console=_quiet_console(),
        )
        result = deployer.deploy(force=True)
        assert result.success

        for target_dir in (".claude", ".opencode", ".agents"):
            assert (
                tmp_path / target_dir / "skills" / "test-skill" / "SKILL.md"
            ).is_file()

    def test_dry_run_creates_no_files(self, tmp_path: Path) -> None:
        skills_dir = _make_skill_tree(tmp_path)
        deployer = SkillDeployer(
            skills_dir=skills_dir,
            project_root=tmp_path,
            console=_quiet_console(),
        )
        result = deployer.deploy(
            targets=[DeployTarget.CLAUDE],
            dry_run=True,
            force=True,
        )
        assert result.dry_run
        assert result.success
        assert len(result.actions) > 0
        assert not (tmp_path / ".claude" / "skills").exists()

    def test_skip_without_force(self, tmp_path: Path) -> None:
        skills_dir = _make_skill_tree(tmp_path)
        deployer = SkillDeployer(
            skills_dir=skills_dir,
            project_root=tmp_path,
            console=_quiet_console(),
        )

        # Deploy once with force
        deployer.deploy(targets=[DeployTarget.CLAUDE], force=True)

        # Deploy again without force - should skip existing
        result = deployer.deploy(targets=[DeployTarget.CLAUDE], force=False)
        assert result.success
        skip_actions = [a for a in result.actions if a.action == "skip"]
        assert len(skip_actions) > 0  # SKILL.md files should be skipped

    def test_overwrite_with_force(self, tmp_path: Path) -> None:
        skills_dir = _make_skill_tree(tmp_path)
        deployer = SkillDeployer(
            skills_dir=skills_dir,
            project_root=tmp_path,
            console=_quiet_console(),
        )

        # Deploy twice with force
        deployer.deploy(targets=[DeployTarget.CLAUDE], force=True)
        result = deployer.deploy(targets=[DeployTarget.CLAUDE], force=True)
        assert result.success
        overwrite_actions = [a for a in result.actions if a.action == "overwrite"]
        assert len(overwrite_actions) > 0

    def test_empty_skills_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        deployer = SkillDeployer(
            skills_dir=empty_dir,
            project_root=tmp_path,
            console=_quiet_console(),
        )
        result = deployer.deploy(targets=[DeployTarget.CLAUDE], force=True)
        assert not result.success
        assert any("No skills found" in e for e in result.errors)

    def test_file_content_preserved(self, tmp_path: Path) -> None:
        """Verify deployed files have the same content as source."""
        skills_dir = _make_skill_tree(tmp_path)
        deployer = SkillDeployer(
            skills_dir=skills_dir,
            project_root=tmp_path,
            console=_quiet_console(),
        )
        deployer.deploy(targets=[DeployTarget.CLAUDE], force=True)

        src = skills_dir / "test-skill" / "SKILL.md"
        dst = tmp_path / ".claude" / "skills" / "test-skill" / "SKILL.md"
        assert src.read_text(encoding="utf-8") == dst.read_text(encoding="utf-8")


# -- DeployTarget tests -----------------------------------------------------


class TestDeployTarget:
    def test_all_returns_three(self) -> None:
        assert len(DeployTarget.all()) == 3

    def test_values(self) -> None:
        assert DeployTarget.CLAUDE.value == "claude"
        assert DeployTarget.OPENCODE.value == "opencode"
        assert DeployTarget.AGENTS.value == "agents"
