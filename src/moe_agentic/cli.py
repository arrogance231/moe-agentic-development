"""CLI for MoE Agentic Skills -- deploy, list, validate, and inspect skills."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from moe_agentic.deploy import DeployTarget, SkillDeployer
from moe_agentic.skill_loader import SkillLoader, validate_skill_name

console = Console()

# -- Shared options ---------------------------------------------------------

_DEFAULT_SKILLS_DIR = "skills"


def _skills_dir_option(func):  # type: ignore[no-untyped-def]
    """Shared --skills-dir option decorator."""
    return click.option(
        "--skills-dir",
        type=click.Path(exists=False, path_type=Path),
        default=_DEFAULT_SKILLS_DIR,
        show_default=True,
        help="Root directory containing skill subdirectories.",
    )(func)


# -- CLI group --------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """MoE Agentic Skills -- deploy and manage agent skills."""


# -- deploy -----------------------------------------------------------------


@cli.command()
@click.option(
    "--target",
    "-t",
    type=click.Choice(
        ["claude", "opencode", "agents", "all"], case_sensitive=False
    ),
    default="all",
    show_default=True,
    help="Deployment target runtime.",
)
@click.option(
    "--global",
    "global_install",
    is_flag=True,
    default=False,
    help="Deploy to user-global directories instead of project-local.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview deployment without writing files.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Overwrite existing files without confirmation.",
)
@_skills_dir_option
def deploy(
    target: str,
    global_install: bool,
    dry_run: bool,
    force: bool,
    skills_dir: Path,
) -> None:
    """Deploy skills to agent runtime directories.

    Copies SKILL.md and optional knowledge/, tools/, examples/ subdirectories
    to the target runtime's skills directory.

    Examples:

        moe-skills deploy --target claude

        moe-skills deploy --target all --dry-run

        moe-skills deploy --global --force
    """
    if not skills_dir.is_dir():
        console.print(f"[red]Skills directory not found:[/] {skills_dir}")
        raise SystemExit(1)

    # Resolve targets
    if target == "all":
        targets = DeployTarget.all()
    else:
        targets = [DeployTarget(target)]

    deployer = SkillDeployer(
        skills_dir=skills_dir,
        console=console,
    )
    result = deployer.deploy(
        targets=targets,
        global_install=global_install,
        dry_run=dry_run,
        force=force,
    )
    if not result.success:
        raise SystemExit(1)


# -- list -------------------------------------------------------------------


@cli.command(name="list")
@_skills_dir_option
def list_skills(skills_dir: Path) -> None:
    """List all discovered skills.

    Shows a table of skill names, descriptions, and available subdirectories.
    """
    if not skills_dir.is_dir():
        console.print(f"[red]Skills directory not found:[/] {skills_dir}")
        raise SystemExit(1)

    loader = SkillLoader(skills_dir=skills_dir)
    try:
        skills = loader.load_all()
    except Exception as exc:
        console.print(f"[red]Error loading skills:[/] {exc}")
        raise SystemExit(1)

    if not skills:
        console.print("[yellow]No skills found.[/]")
        return

    table = Table(title=f"Skills in {skills_dir}", show_lines=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Subdirs", style="dim")

    for skill in skills.values():
        subdirs = ", ".join(skill.subdirectories) or "-"
        table.add_row(skill.name, skill.metadata.description, subdirs)

    console.print(table)
    console.print(f"\n[dim]{len(skills)} skill(s) found.[/]")


# -- validate ---------------------------------------------------------------


@cli.command()
@_skills_dir_option
def validate(skills_dir: Path) -> None:
    """Validate all skills in the directory.

    Checks frontmatter, naming conventions, and structure.
    """
    if not skills_dir.is_dir():
        console.print(f"[red]Skills directory not found:[/] {skills_dir}")
        raise SystemExit(1)

    loader = SkillLoader(skills_dir=skills_dir)
    issues = loader.validate_all()

    # Also count total discovered (including invalid names)
    total = len(loader.discover_all())
    failed = len(issues)
    passed = total - failed

    if issues:
        for dir_name, errors in sorted(issues.items()):
            console.print(f"\n[bold red]\u2718 {dir_name}[/]")
            for err in errors:
                console.print(f"  [red]- {err}[/]")

    console.print(
        f"\n[bold]{total}[/] skill(s) checked: "
        f"[green]{passed} passed[/], [red]{failed} failed[/]"
    )

    if issues:
        raise SystemExit(1)
    else:
        console.print("[bold green]\u2714 All skills are valid.[/]")


# -- info -------------------------------------------------------------------


@cli.command()
@click.argument("skill_name")
@_skills_dir_option
def info(skill_name: str, skills_dir: Path) -> None:
    """Show detailed information about a specific skill.

    SKILL_NAME is the name (directory) of the skill to inspect.
    """
    if not skills_dir.is_dir():
        console.print(f"[red]Skills directory not found:[/] {skills_dir}")
        raise SystemExit(1)

    loader = SkillLoader(skills_dir=skills_dir)
    skill = loader.load_by_name(skill_name)

    if skill is None:
        console.print(f"[red]Skill not found:[/] {skill_name}")
        all_skills = loader.load_all()
        if all_skills:
            available = [s.name for s in all_skills.values()]
            console.print(f"[dim]Available: {', '.join(available)}[/]")
        raise SystemExit(1)

    # Name validation
    name_errors = validate_skill_name(skill.name)
    valid_badge = (
        "[green]\u2714 valid[/]"
        if not name_errors
        else "[red]\u2718 invalid[/]"
    )

    # Build info panel
    lines = [
        f"[bold]Name:[/]        {skill.name}  {valid_badge}",
        f"[bold]Description:[/] {skill.metadata.description}",
    ]
    if skill.metadata.argument_hint:
        lines.append(
            f"[bold]Argument:[/]    {skill.metadata.argument_hint}"
        )
    if skill.metadata.license:
        lines.append(f"[bold]License:[/]     {skill.metadata.license}")
    if skill.metadata.compatibility:
        lines.append(
            f"[bold]Compat:[/]      "
            f"{', '.join(skill.metadata.compatibility)}"
        )

    lines.append(f"[bold]Source:[/]      {skill.source_dir}")
    lines.append(
        f"[bold]Subdirs:[/]     {', '.join(skill.subdirectories) or 'none'}"
    )

    # Body preview (first 5 non-empty lines)
    body_lines = [
        bl for bl in skill.body.strip().splitlines() if bl.strip()
    ][:5]
    if body_lines:
        lines.append("")
        lines.append("[bold]Instructions preview:[/]")
        for bl in body_lines:
            lines.append(f"  [dim]{bl[:120]}[/]")

    if name_errors:
        lines.append("")
        lines.append("[bold red]Validation issues:[/]")
        for err in name_errors:
            lines.append(f"  [red]- {err}[/]")

    if skill.metadata.extra:
        lines.append("")
        lines.append(f"[bold]Extra fields:[/] {skill.metadata.extra}")

    panel = Panel(
        "\n".join(lines),
        title=f"Skill: {skill.name}",
        border_style="cyan",
    )
    console.print(panel)


if __name__ == "__main__":
    cli()
