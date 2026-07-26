"""Running one task N times from an identical starting tree."""

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("eqty.lineage.replicate")

DEFAULT_AGENT = "claude"


@dataclass
class RunSpec:
    """What to run, N times."""

    task: str
    template: Path
    """A working tree copied fresh for each run. Every run must start from identical bytes, or the
    comparison measures the starting state rather than the agent."""
    n: int = 6
    model: Optional[str] = None
    max_turns: int = 20
    permission_mode: str = "acceptEdits"
    agent: str = DEFAULT_AGENT
    timeout: int = 900


@dataclass
class ReplicationResult:
    spec: RunSpec
    run_dirs: Dict[str, Path] = field(default_factory=dict)
    transcripts: Dict[str, Path] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return len(self.transcripts) >= 2

    def summary(self) -> str:
        return (
            f"{len(self.transcripts)}/{self.spec.n} runs captured in {self.elapsed:.0f}s"
            + (f"  ({len(self.failures)} failed)" if self.failures else "")
        )


def transcript_for(run_dir: Path, root: Optional[Path] = None) -> Optional[Path]:
    """The session transcript the agent wrote for this working directory.

    Claude Code names a project directory after the slugified cwd, so the run directory determines
    where its transcript lands. The newest file wins: a directory reused across experiments accumulates
    sessions, and the one just written is the one we mean.
    """
    base = root or (Path.home() / ".claude" / "projects")
    slug = str(run_dir.resolve()).replace("/", "-")
    project = base / slug
    if not project.is_dir():
        return None
    sessions = sorted(project.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0] if sessions else None


def _command(spec: RunSpec) -> List[str]:
    cmd = [spec.agent, "-p", spec.task, "--permission-mode", spec.permission_mode]
    if spec.model:
        cmd += ["--model", spec.model]
    if spec.max_turns:
        cmd += ["--max-turns", str(spec.max_turns)]
    return cmd


def replicate(spec: RunSpec, workspace: Path, on_run=None) -> ReplicationResult:
    """Run the task ``spec.n`` times, each in its own copy of the template tree.

    Runs are sequential on purpose. Running them concurrently would share CPU, disk and rate limits in
    ways that could correlate the outcomes -- and the entire measurement is about whether outcomes are
    independent.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    result = ReplicationResult(spec=spec)
    started = time.time()

    for i in range(1, spec.n + 1):
        name = f"run-{i}"
        run_dir = workspace / name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        shutil.copytree(spec.template, run_dir)
        # a copied .git would let the agent see (and act on) the template's history
        shutil.rmtree(run_dir / ".git", ignore_errors=True)

        try:
            subprocess.run(
                _command(spec),
                cwd=run_dir,
                capture_output=True,
                timeout=spec.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result.failures.append(f"{name}: {exc}")
            logger.warning("run %s failed: %s", name, exc)
            continue

        transcript = transcript_for(run_dir)
        if transcript is None:
            result.failures.append(f"{name}: no transcript found")
            continue

        result.run_dirs[name] = run_dir
        result.transcripts[name] = transcript
        if on_run:
            on_run(name, run_dir, transcript)

    result.elapsed = time.time() - started
    return result


__all__ = ["DEFAULT_AGENT", "ReplicationResult", "RunSpec", "replicate", "transcript_for"]
