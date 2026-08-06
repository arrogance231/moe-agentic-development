"""Tests for the CLI."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from moe_agentic.cli import cli


_SKILL_MD = textwrap.dedent("""\
    ---
    name: demo-skill
    description: A demo skill for CLI testing.
    argument-hint: <demo-arg>
    ---

    # Demo Skill

    You are a demo. Do demo things.
""")


def _setup_skills(tmp_path: Path) -> Path:
    skills_dir = tmp_path / "skills"
    s1 = skills_dir / "demo-skill"
    s1.mkdir(parents=True)
    (s1 / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    (s1 / "knowledge").mkdir()
    (s1 / "knowledge" / "ref.md").write_text("# ref", encoding="utf-8")
    return skills_dir


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestDeployCommand:
    def test_deploy_claude(self, runner: CliRunner, tmp_path: Path) -> None:
        skills_dir = _setup_skills(tmp_path)
        result = runner.invoke(
            cli,
            ["deploy", "--target", "claude", "--force", "--skills-dir", str(skills_dir)],
        )
        assert result.exit_code == 0, result.output
        # Should have created the file in CWD-relative .claude/skills
        # (we can't fully control CWD in test, so just check output)
        assert "deployed" in result.output.lower() or "Deployment" in result.output

    def test_deploy_dry_run(self, runner: CliRunner, tmp_path: Path) -> None:
        skills_dir = _setup_skills(tmp_path)
        result = runner.invoke(
            cli,
            ["deploy", "--target", "claude", "--dry-run", "--skills-dir", str(skills_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output

    def test_deploy_missing_dir(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            ["deploy", "--skills-dir", "/nonexistent/path/xyz"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


class TestListCommand:
    def test_list_skills(self, runner: CliRunner, tmp_path: Path) -> None:
        skills_dir = _setup_skills(tmp_path)
        result = runner.invoke(cli, ["list", "--skills-dir", str(skills_dir)])
        assert result.exit_code == 0, result.output
        assert "demo-skill" in result.output

    def test_list_empty(self, runner: CliRunner, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(cli, ["list", "--skills-dir", str(empty)])
        assert result.exit_code == 0
        assert "No skills found" in result.output


class TestValidateCommand:
    def test_validate_valid(self, runner: CliRunner, tmp_path: Path) -> None:
        skills_dir = _setup_skills(tmp_path)
        result = runner.invoke(cli, ["validate", "--skills-dir", str(skills_dir)])
        assert result.exit_code == 0, result.output
        assert "1 passed" in result.output or "All skills are valid" in result.output

    def test_validate_invalid(self, runner: CliRunner, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        bad = skills_dir / "BAD_NAME"
        bad.mkdir(parents=True)
        (bad / "SKILL.md").write_text(
            "---\nname: BAD_NAME\ndescription: bad\n---\nstuff\n",
            encoding="utf-8",
        )
        result = runner.invoke(cli, ["validate", "--skills-dir", str(skills_dir)])
        assert result.exit_code != 0


class TestInfoCommand:
    def test_info_found(self, runner: CliRunner, tmp_path: Path) -> None:
        skills_dir = _setup_skills(tmp_path)
        result = runner.invoke(cli, ["info", "demo-skill", "--skills-dir", str(skills_dir)])
        assert result.exit_code == 0, result.output
        assert "demo-skill" in result.output
        assert "A demo skill for CLI testing" in result.output

    def test_info_not_found(self, runner: CliRunner, tmp_path: Path) -> None:
        skills_dir = _setup_skills(tmp_path)
        result = runner.invoke(cli, ["info", "nonexistent", "--skills-dir", str(skills_dir)])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
