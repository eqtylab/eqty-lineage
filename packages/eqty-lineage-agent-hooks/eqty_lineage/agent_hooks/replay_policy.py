"""Counterfactual policy replay: what would this session have produced under a different policy?

A permission hook is a **handler** in the algebraic-effects sense -- it interprets a tool operation and
says allow, deny, or defer. A recorded trace is then a term over those operations, and reinterpreting it
under a different handler is exactly what "what if the policy had been stricter" means.

The handler used here is the *same object* that runs live: :class:`~.policy.HookPolicy`, given a
synthesised payload. A reimplementation would let the counterfactual and the enforcement drift apart,
which would make the answer worse than useless.

**What replay can and cannot tell you.** Denying a tool call changes what the model would have done
next, and that is unknowable without re-running. So replay reports *which recorded artifacts become
unreachable* -- never *what the agent would have done instead*. Every function here is named for the
former. If you want the latter, replicate under the policy and compare; that is a different, more
expensive question.

Replaying under K policies gives a determination space over **policies** rather than runs, so the same
supports, ``qdepth`` and robustness queries apply unchanged -- one algebra, two instantiations.
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from eqty_lineage.core import prov

from .policy import HookPolicy

logger = logging.getLogger("eqty.lineage.hooks")


@dataclass
class PolicyOutcome:
    """What a single policy would have cost this session."""

    policy: str
    denied: Set[str] = field(default_factory=set)
    """Activities the handler would have refused."""
    lost: Set[str] = field(default_factory=set)
    """Entities that become unreachable: the denied outputs and everything downstream of them."""
    lost_paths: List[str] = field(default_factory=list)
    surviving: Set[str] = field(default_factory=set)

    def summary(self) -> str:
        if not self.denied:
            return f"{self.policy}: no tool call refused; the session is unchanged"
        return (
            f"{self.policy}: {len(self.denied)} call(s) refused, "
            f"{len(self.lost)} entities unreachable"
            + (f" ({len(self.lost_paths)} file versions)" if self.lost_paths else "")
        )


def _index(triples) -> Tuple[Dict, Dict, Dict, Dict]:
    paths, labels = {}, {}
    generated: Dict[str, str] = {}
    used: Dict[str, List[str]] = {}
    for t in triples:
        if t.predicate == prov.HAS_PATH:
            paths[t.subject] = t.object
        elif t.predicate == prov.LABEL:
            labels[t.subject] = t.object
        elif t.predicate == prov.WAS_GENERATED_BY:
            generated[t.subject] = t.object
        elif t.predicate == prov.USED:
            used.setdefault(t.subject, []).append(t.object)
    return paths, labels, generated, used


def replay(triples, policy: HookPolicy, name: str = "policy") -> PolicyOutcome:
    """Reinterpret a recorded trace under ``policy``.

    An activity is refused when the handler denies the write it performed. Everything that activity
    produced then never exists, and neither does anything derived from it -- that cascade is ordinary
    reachability, computed over the same graph the manifest is built from.
    """
    triples = list(triples)
    paths, labels, generated, used = _index(triples)

    # Which activities would have been refused? Ask the live handler, with a payload shaped exactly as
    # the hook would have delivered it, so the counterfactual cannot diverge from enforcement.
    denied: Set[str] = set()
    for entity, activity in generated.items():
        path = paths.get(entity)
        if path is None:
            continue
        decision = policy.decide(
            {
                "tool_name": labels.get(activity, "Edit"),
                "tool_input": {"file_path": path},
                "permission_mode": "replay",
            }
        )
        if decision is not None and decision[0] == "deny":
            denied.add(activity)

    # Everything a refused activity produced, and everything downstream of that.
    produced_by_denied = {e for e, a in generated.items() if a in denied}
    lost = set(produced_by_denied)
    frontier = list(produced_by_denied)
    consumers: Dict[str, List[str]] = {}
    for activity, inputs in used.items():
        for i in inputs:
            consumers.setdefault(i, []).append(activity)

    # activity -> what it produced, built once. Scanning `generated` inside the walk would make the
    # cascade O(entities) per step and quadratic overall.
    produces: Dict[str, List[str]] = {}
    for entity, activity in generated.items():
        produces.setdefault(activity, []).append(entity)

    while frontier:
        entity = frontier.pop()
        for activity in consumers.get(entity, ()):
            for downstream in produces.get(activity, ()):
                if downstream not in lost:
                    lost.add(downstream)
                    frontier.append(downstream)

    entities = set(paths) | set(generated)
    return PolicyOutcome(
        policy=name,
        denied=denied,
        lost=lost,
        lost_paths=sorted({paths[e] for e in lost if e in paths}),
        surviving=entities - lost,
    )


@dataclass
class PolicyComparison:
    """Determination over policies: which artifacts survive every policy considered."""

    policies: List[str]
    outcomes: Dict[str, PolicyOutcome]
    support: Dict[str, int] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)

    @property
    def all_policies(self) -> int:
        return (1 << len(self.policies)) - 1

    def robust(self) -> Set[str]:
        """Artifacts that survive under every policy -- full support, ``qdepth`` 0."""
        return {e for e, m in self.support.items() if m == self.all_policies}

    def contingent(self) -> Dict[str, List[str]]:
        """Artifacts that exist only under some policies, mapped to those policies."""
        out = {}
        for entity, mask in self.support.items():
            if mask != self.all_policies and mask:
                out[self.paths.get(entity, entity)] = [
                    p for i, p in enumerate(self.policies) if mask >> i & 1
                ]
        return out

    def render(self) -> str:
        lines = [o.summary() for o in self.outcomes.values()]
        robust = self.robust()
        contingent = self.contingent()
        lines += [
            "",
            f"{len(robust)} artifacts survive every policy; {len(contingent)} are policy-contingent.",
        ]
        for path, policies in sorted(contingent.items())[:10]:
            lines.append(f"    {path}  survives only under: {', '.join(policies)}")
        lines += [
            "",
            "Replay reports which *recorded* artifacts become unreachable. It cannot say what the agent",
            "would have done instead -- that needs a re-run under the policy, not a reinterpretation.",
        ]
        return "\n".join(lines)


def compare_policies(triples, policies: Mapping[str, HookPolicy]) -> PolicyComparison:
    """Replay one trace under several policies and take the determination over them.

    The resulting supports are the same structure :mod:`eqty_lineage.query.determination` computes over
    runs; only the resolution space differs. An artifact with full support is invariant to the
    governance change, one with partial support exists only because a particular policy allowed it.
    """
    names = list(policies)
    outcomes = {name: replay(triples, policies[name], name) for name in names}

    paths, _labels, generated, _used = _index(list(triples))
    entities = set(paths) | set(generated)

    support: Dict[str, int] = {}
    for i, name in enumerate(names):
        for entity in outcomes[name].surviving & entities:
            support[entity] = support.get(entity, 0) | (1 << i)

    return PolicyComparison(policies=names, outcomes=outcomes, support=support, paths=paths)


__all__ = ["PolicyComparison", "PolicyOutcome", "compare_policies", "replay"]
