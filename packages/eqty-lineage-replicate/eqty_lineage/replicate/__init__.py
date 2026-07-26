"""Run an agent task N times and report which artifacts survived the model's nondeterminism.

A signature proves you got *one* outcome. It does not tell you which, or whether another run would have
produced something else. Replication is the only way to find out, and content-addressed lineage makes
the comparison exact: identical bytes get identical CIDs, so runs join automatically with no correlation
ids and no shared database.

**This costs N times the tokens.** That is the honest trade for knowing whether an artifact is stable,
and it belongs in the decision to run it, not in a footnote.

Measured on a task specified down to the algorithm, with the test's literal assertion dictated: six runs
produced five distinct implementations, three behaviourally distinct, all passing the dictated test. You
cannot prompt your way to reproducible artifacts.
"""

from .driver import ReplicationResult, RunSpec, replicate
from .report import RobustnessReport, build_report

__all__ = [
    "ReplicationResult",
    "RobustnessReport",
    "RunSpec",
    "build_report",
    "replicate",
]
