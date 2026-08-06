"""Skill deployment to agent runtime directories.

Deploys skills from a source directory to Claude Code, OpenCode,
and generic .agents/ runtime directories. Supports project-local
and user-global installation scopes with atomic overwrites.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.table import Table

from moe_agentic.skill_loader import Skill, SkillLoader


class DeployTarget(Enum):
    """Supported agent runtime deployment targets."""

    CLAUDE = "claude"
    OPENCODE = "opencode"
    AGENTS = "agents"

    @classmethod
    def all(cls) -> list[DeployTarget]:
        """Return all deployment targets."""
        return list(cls)


# -- Path resolution --------------------------------------------------------


def _project_skill_dir(target: DeployTarget, project_root: Path) -> Path:
    """Return the project-local skills directory for a target.

    Args:
        target: The deployment target runtime.
        project_root: Root of the project.

    Returns:
        Path to the project-local skills directory.
    """
    mapping = {
        DeployTarget.CLAUDE: project_root / ".claude" / "skills",
        DeployTarget.OPENCODE: project_root / ".opencode" / "skills",
        DeployTarget.AGENTS: project_root / ".agents" / "skills",
    }
    return mapping[target]


def _global_skill_dir(target: DeployTarget) -> Path:
    """Return the global (user-level) skills directory for a target.

    Args:
        target: The deployment target runtime.

    Returns:
        Path to the user-global skills directory.
    """
    home = Path.home()
    mapping = {
        DeployTarget.CLAUDE: home / ".claude" / "skills",
        DeployTarget.OPENCODE: home / ".config" / "opencode" / "skills",
        DeployTarget.AGENTS: home / ".agents" / "skills",
    }
    return mapping[target]


def resolve_skills_dir(
    target: DeployTarget,
    *,
    global_install: bool = False,
    project_root: Path | None = None,
) -> Path:
    """Resolve the destination skills directory for a target and scope.

    Args:
        target: The deployment target runtime.
        global_install: If True, use user-global directory.
        project_root: Project root for project-local installs (defaults to cwd).

    Returns:
        Resolved destination path.
    """
    if global_install:
        return _global_skill_dir(target)
    root = project_root or Path.cwd()
    return _project_skill_dir(target, root)


# -- Deployment action log --------------------------------------------------


@dataclass
class DeployAction:
    """Record of a single deployment action.

    Attributes:
        skill_name: Name of the skill being deployed.
        target: The deployment target runtime.
        source: Source path of the file/directory.
        destination: Destination path of the file/directory.
        action: One of 'copy', 'mkdir', 'skip', 'overwrite'.
        is_directory: True if the action involves a directory.
    """

    skill_name: str
    target: DeployTarget
    source: Path
    destination: Path
    action: str
    is_directory: bool = False

    def __str__(self) -> str:
        verb = self.action.upper()
        kind = "dir " if self.is_directory else "file"
        return f"[{verb:>9}] {kind} {self.source} -> {self.destination}"


@dataclass
class DeployResult:
    """Aggregate result of a deployment run.

    Attributes:
        actions: List of deployment actions taken.
        errors: List of error messages.
        dry_run: True if this was a dry-run (no files written).
    """

    actions: list[DeployAction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def success(self) -> bool:
        """True if no errors occurred."""
        return len(self.errors) == 0

    @property
    def skills_deployed(self) -> set[str]:
        """Set of skill names that were deployed."""
        return {
            a.skill_name
            for a in self.actions
            if a.action in ("copy", "overwrite", "mkdir")
        }


# -- Deployer ---------------------------------------------------------------


class SkillDeployer:
    """Deploys skills from a source directory to agent runtime directories.

    Supports Claude Code, OpenCode, and generic .agents/ runtimes.
    Handles project-local and user-global installation scopes.
    Uses atomic copy-to-temp-then-rename to prevent data loss on overwrite.

    Example::

        deployer = SkillDeployer(skills_dir=Path("skills"))
        result = deployer.deploy(
            targets=[DeployTarget.CLAUDE, DeployTarget.OPENCODE],
            dry_run=False,
            force=True,
        )
    """

    #: Subdirectories that are copied alongside SKILL.md.
    COPYABLE_SUBDIRS: tuple[str, ...] = ("knowledge", "tools", "examples")

    def __init__(
        self,
        skills_dir: Path,
        *,
        project_root: Path | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialize the deployer.

        Args:
            skills_dir: Source directory containing skill subdirectories.
            project_root: Project root for project-local deploys (defaults to cwd).
            console: Rich console for output (created if not provided).
        """
        self.skills_dir = skills_dir
        self.project_root = project_root or Path.cwd()
        self.console = console or Console()
        self._loader = SkillLoader(skills_dir=skills_dir)

    # -- public API ---------------------------------------------------------

    def deploy(
        self,
        targets: Sequence[DeployTarget] | None = None,
        *,
        global_install: bool = False,
        dry_run: bool = False,
        force: bool = False,
        skill_names: Sequence[str] | None = None,
    ) -> DeployResult:
        """Deploy skills to one or more runtime targets.

        Args:
            targets: Runtime targets (defaults to all).
            global_install: Deploy to user-global directories.
            dry_run: Preview actions without writing.
            force: Overwrite existing files without prompting.
            skill_names: Optional filter -- deploy only these skills.

        Returns:
            DeployResult with actions taken and any errors.
        """
        if targets is None:
            targets = DeployTarget.all()

        result = DeployResult(dry_run=dry_run)

        # Load skills
        try:
            skills = list(self._loader.load_all().values())
        except Exception as exc:
            result.errors.append(f"Failed to load skills: {exc}")
            return result

        if not skills:
            result.errors.append(f"No skills found in {self.skills_dir}")
            return result

        # Filter by name if requested
        if skill_names:
            name_set = set(skill_names)
            skills = [s for s in skills if s.name in name_set]
            missing = name_set - {s.name for s in skills}
            if missing:
                result.errors.append(
                    f"Skills not found: {', '.join(sorted(missing))}"
                )
                return result

        # Deploy each skill to each target
        for target in targets:
            dest_root = resolve_skills_dir(
                target,
                global_install=global_install,
                project_root=self.project_root,
            )
            for skill in skills:
                self._deploy_skill(
                    skill=skill,
                    target=target,
                    dest_root=dest_root,
                    dry_run=dry_run,
                    force=force,
                    result=result,
                )

        self._print_summary(result)
        return result

    # -- internals ----------------------------------------------------------

    def _deploy_skill(
        self,
        *,
        skill: Skill,
        target: DeployTarget,
        dest_root: Path,
        dry_run: bool,
        force: bool,
        result: DeployResult,
    ) -> None:
        """Deploy a single skill to a single target directory.

        Args:
            skill: The skill to deploy.
            target: The deployment target.
            dest_root: Root skills directory for the target.
            dry_run: If True, record actions but don't write.
            force: If True, overwrite without confirmation.
            result: Accumulator for actions and errors.
        """
        skill_dest = dest_root / skill.name

        # Create destination directory
        action_mkdir = DeployAction(
            skill_name=skill.name,
            target=target,
            source=skill.source_dir,
            destination=skill_dest,
            action="mkdir",
            is_directory=True,
        )
        result.actions.append(action_mkdir)
        if not dry_run:
            try:
                skill_dest.mkdir(parents=True, exist_ok=True)
            except PermissionError as exc:
                result.errors.append(
                    f"Permission denied creating {skill_dest}: {exc}"
                )
                return
            except OSError as exc:
                result.errors.append(
                    f"OS error creating {skill_dest}: {exc}"
                )
                return

        # Copy SKILL.md
        self._copy_file(
            src=skill.skill_md_path,
            dst=skill_dest / "SKILL.md",
            skill_name=skill.name,
            target=target,
            dry_run=dry_run,
            force=force,
            result=result,
        )

        # Copy optional subdirectories
        for subdir_name in self.COPYABLE_SUBDIRS:
            src_subdir = skill.source_dir / subdir_name
            if src_subdir.is_dir():
                dst_subdir = skill_dest / subdir_name
                self._copy_directory(
                    src=src_subdir,
                    dst=dst_subdir,
                    skill_name=skill.name,
                    target=target,
                    dry_run=dry_run,
                    force=force,
                    result=result,
                )

    def _copy_file(
        self,
        *,
        src: Path,
        dst: Path,
        skill_name: str,
        target: DeployTarget,
        dry_run: bool,
        force: bool,
        result: DeployResult,
    ) -> None:
        """Copy a single file with atomic overwrite.

        Uses copy-to-temp-then-rename to prevent data loss.  If the
        destination exists and *force* is ``False``, the copy is skipped.

        Args:
            src: Source file path.
            dst: Destination file path.
            skill_name: Name of the skill being deployed.
            target: Deployment target.
            dry_run: Preview only.
            force: Overwrite existing.
            result: Accumulator.
        """
        if dst.exists() and not force:
            action = DeployAction(
                skill_name=skill_name,
                target=target,
                source=src,
                destination=dst,
                action="skip",
            )
            result.actions.append(action)
            return

        action_type = "overwrite" if dst.exists() else "copy"
        action = DeployAction(
            skill_name=skill_name,
            target=target,
            source=src,
            destination=dst,
            action=action_type,
        )
        result.actions.append(action)

        if not dry_run:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                # Atomic overwrite: copy to temp file in same dir, then rename
                fd, tmp_path = tempfile.mkstemp(
                    dir=dst.parent, suffix=".tmp"
                )
                try:
                    import os

                    os.close(fd)
                    shutil.copy2(src, tmp_path)
                    Path(tmp_path).replace(dst)
                except BaseException:
                    # Clean up temp file on any failure
                    Path(tmp_path).unlink(missing_ok=True)
                    raise
            except PermissionError as exc:
                result.errors.append(
                    f"Permission denied copying {src} -> {dst}: {exc}"
                )
            except OSError as exc:
                result.errors.append(
                    f"OS error copying {src} -> {dst}: {exc}"
                )

    def _copy_directory(
        self,
        *,
        src: Path,
        dst: Path,
        skill_name: str,
        target: DeployTarget,
        dry_run: bool,
        force: bool,
        result: DeployResult,
    ) -> None:
        """Recursively copy a directory with atomic overwrite.

        When overwriting, copies to a temp directory first, then removes
        the old directory and renames the temp into place.

        Args:
            src: Source directory path.
            dst: Destination directory path.
            skill_name: Name of the skill being deployed.
            target: Deployment target.
            dry_run: Preview only.
            force: Overwrite existing.
            result: Accumulator.
        """
        if dst.exists() and not force:
            action = DeployAction(
                skill_name=skill_name,
                target=target,
                source=src,
                destination=dst,
                action="skip",
                is_directory=True,
            )
            result.actions.append(action)
            return

        action_type = "overwrite" if dst.exists() else "copy"
        dir_action = DeployAction(
            skill_name=skill_name,
            target=target,
            source=src,
            destination=dst,
            action=action_type,
            is_directory=True,
        )
        result.actions.append(dir_action)

        if not dry_run:
            try:
                # Atomic directory overwrite: copy to temp, swap
                tmp_dst = dst.with_name(dst.name + ".tmp")
                if tmp_dst.exists():
                    shutil.rmtree(tmp_dst)
                shutil.copytree(src, tmp_dst)
                if dst.exists():
                    shutil.rmtree(dst)
                tmp_dst.rename(dst)
            except PermissionError as exc:
                result.errors.append(
                    f"Permission denied copying dir {src} -> {dst}: {exc}"
                )
                # Clean up partial temp
                if tmp_dst.exists():
                    shutil.rmtree(tmp_dst, ignore_errors=True)
            except OSError as exc:
                result.errors.append(
                    f"OS error copying dir {src} -> {dst}: {exc}"
                )
                if tmp_dst.exists():
                    shutil.rmtree(tmp_dst, ignore_errors=True)

    def _print_summary(self, result: DeployResult) -> None:
        """Print a rich summary of the deployment result.

        Args:
            result: The deployment result to summarize.
        """
        if result.dry_run:
            self.console.print(
                "\n[bold yellow]DRY RUN[/] - no files were modified.\n"
            )

        if result.actions:
            table = Table(title="Deployment Actions", show_lines=False)
            table.add_column("Action", style="bold", width=10)
            table.add_column("Skill", style="cyan")
            table.add_column("Target", style="magenta")
            table.add_column("Destination", style="dim")

            for a in result.actions:
                style_map = {
                    "copy": "green",
                    "mkdir": "blue",
                    "overwrite": "yellow",
                    "skip": "dim",
                }
                style = style_map.get(a.action, "")
                table.add_row(
                    f"[{style}]{a.action}[/{style}]",
                    a.skill_name,
                    a.target.value,
                    str(a.destination),
                )
            self.console.print(table)

        if result.errors:
            self.console.print("\n[bold red]Errors:[/]")
            for err in result.errors:
                self.console.print(f"  [red]\u2718[/] {err}")
        else:
            deployed = result.skills_deployed
            label = "would be deployed" if result.dry_run else "deployed"
            self.console.print(
                f"\n[bold green]\u2714[/] {len(deployed)} skill(s) {label}"
                " successfully."
            )
