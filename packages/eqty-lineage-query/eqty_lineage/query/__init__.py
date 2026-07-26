"""Semiring-parameterized Datalog over EQTY lineage graphs.

One evaluator, instantiated at different semirings, answers structurally different questions over the
same fact set: whether something is reachable, how many ways, by which minimal sets of facts, and under
what taint.

    from eqty_lineage.query import blast_radius, taint, ABSORPTIVE

    downstream = blast_radius(triples, source_cid, ABSORPTIVE)
"""

from .accel import AVAILABLE as RUST_AVAILABLE, BackendUnavailable
from .determination import (
    MultiRun,
    PathVerdict,
    divergence_report,
    influence_support,
    qdepth,
    robust,
    support,
    union_runs,
)
from .engine import (
    Atom,
    Database,
    NonTerminating,
    Rule,
    Var,
    annotate,
    atom,
    edb_from_triples,
    evaluate,
    rule,
)
from .queries import (
    INFLUENCE_RULES,
    VIOLATION_RULES,
    AlternativesReport,
    QueryResult,
    blast_radius,
    measure_alternatives,
    policy_violations,
    reaches,
    taint,
)
from .semiring import (
    ABSORPTIVE,
    BOOLEAN,
    COUNTING,
    WHY,
    TROPICAL,
    Level,
    Semiring,
    confidentiality,
    integrity,
)

__all__ = [
    "ABSORPTIVE",
    "BOOLEAN",
    "COUNTING",
    "INFLUENCE_RULES",
    "TROPICAL",
    "VIOLATION_RULES",
    "WHY",
    "RUST_AVAILABLE",
    "AlternativesReport",
    "MultiRun",
    "PathVerdict",
    "BackendUnavailable",
    "Atom",
    "Database",
    "Level",
    "NonTerminating",
    "QueryResult",
    "Rule",
    "Semiring",
    "Var",
    "annotate",
    "atom",
    "blast_radius",
    "divergence_report",
    "confidentiality",
    "edb_from_triples",
    "evaluate",
    "influence_support",
    "integrity",
    "measure_alternatives",
    "policy_violations",
    "qdepth",
    "reaches",
    "robust",
    "support",
    "union_runs",
    "rule",
    "taint",
]
