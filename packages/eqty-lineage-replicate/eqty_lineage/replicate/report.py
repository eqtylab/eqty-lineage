"""Turning N captured runs into a robustness verdict."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

logger = logging.getLogger("eqty.lineage.replicate")


@dataclass
class RobustnessReport:
    runs: List[str]
    robust_paths: List[str] = field(default_factory=list)
    divergent: List = field(default_factory=list)  # List[PathVerdict]
    multi: object = None

    @property
    def total_paths(self) -> int:
        return len(self.robust_paths) + len(self.divergent)

    @property
    def fraction_robust(self) -> float:
        return len(self.robust_paths) / self.total_paths if self.total_paths else 1.0

    def render(self, color: bool = True) -> str:
        green, red, dim, reset = ("\033[32m", "\033[31m", "\033[90m", "\033[0m") if color else ("",) * 4
        out = [
            f"{len(self.runs)} runs   "
            f"{len(self.robust_paths)}/{self.total_paths} paths robust "
            f"({100 * self.fraction_robust:.0f}%)",
            "",
        ]
        for verdict in self.divergent:
            out.append(f"{red}DIVERGENT{reset}  {verdict.path}   {verdict.distinct} distinct contents")
            groups = sorted(verdict.contents.items(), key=lambda kv: -self.multi.count(kv[1]))
            for _cid, mask in groups:
                names = ", ".join(self.multi.names(mask))
                out.append(f"    {dim}{self.multi.count(mask)} run(s):{reset} {names}")
        for path in self.robust_paths:
            out.append(f"{green}robust{reset}     {path}")

        if not self.divergent:
            out += ["", "Every path agreed across every run. Nothing here depends on a resolution."]
        else:
            out += [
                "",
                "A divergent path means the runs ended in different bytes. A signature over any one of",
                "them proves you got that one -- not that it was the only possible outcome.",
            ]
        return "\n".join(out)


def build_report(
    transcripts: Mapping[str, Path],
    run_dirs: Mapping[str, Path],
    only: Optional[str] = None,
) -> RobustnessReport:
    """Ingest each run's transcript, union them, and classify every path.

    The SDK is imported lazily so a caller can drive replication without a signer configured; only the
    report needs lineage.
    """
    import os
    import tempfile

    from eqty_lineage.core import LineageRecorder, TripleSink
    from eqty_lineage.query.determination import divergence_report, union_runs
    from eqty_lineage.transcript.claude_code import ClaudeCodeTranscript

    # Resolve every path BEFORE changing directory. run_dirs may hold relative paths, and resolving
    # them from inside the temp working directory silently yields nonexistent roots -- which makes the
    # prefix strip a no-op and every path look unique to its run, i.e. "100% robust" for runs that in
    # fact disagreed.
    roots: Dict[str, str] = {name: str(Path(d).resolve()) for name, d in run_dirs.items()}
    transcripts = {name: Path(t).resolve() for name, t in transcripts.items()}

    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp(prefix="eqty-replicate-"))
    try:
        from eqty_sdk import Context, Signer, init, set_active_signer
        from eqty_sdk.context import graph_context

        init().set_store_all_blobs(False)
        set_active_signer(Signer.new(name="eqty-lineage-replicate", _load_if_exists=True))

        sinks: Dict[str, TripleSink] = {}
        for name, transcript in transcripts.items():
            sink = TripleSink()
            with graph_context(Context.new(name)):
                LineageRecorder(triples=sink, framework="claude-code").handle_all(
                    list(ClaudeCodeTranscript(Path(transcript)).events())
                )
            sinks[name] = sink
    finally:
        os.chdir(cwd)

    multi = union_runs(sinks, roots)
    verdicts = divergence_report(multi)
    if only:
        verdicts = [v for v in verdicts if v.path.endswith(only)]

    return RobustnessReport(
        runs=multi.runs,
        robust_paths=[v.path for v in verdicts if v.is_robust],
        divergent=[v for v in verdicts if not v.is_robust],
        multi=multi,
    )


__all__ = ["RobustnessReport", "build_report"]
