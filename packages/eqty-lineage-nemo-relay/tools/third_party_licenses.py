"""Collect the license text of every crate linked into the plugin.

The cdylib statically links its whole dependency tree, so the bundle a user installs carries
those crates' code and owes their notices. MPL-2.0 and BSD terms make that an obligation rather
than a courtesy.

Only normal dependencies are walked. Dev-dependencies build the tests and build-dependencies
run at compile time; neither reaches the cdylib, and attributing them would claim the artifact
carries code it does not.

Texts come from the crate sources cargo already unpacked, so this needs no network and no
extra tool. Identical texts are emitted once: Apache-2.0 is byte-identical everywhere, while
each MIT notice carries its own copyright line and stays distinct.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

LICENSE_FILES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "UNLICENSE")
OURS = {"eqty-lineage-nemo-relay", "eqty-lineage-nemo-relay-abi-test"}


def texts(crate_dir):
    """Every license-ish file in a crate's source root, longest first."""
    found = []
    for path in sorted(crate_dir.iterdir()) if crate_dir.is_dir() else []:
        if not path.is_file():
            continue
        stem = path.name.upper()
        if any(stem.startswith(p) for p in LICENSE_FILES):
            try:
                found.append((path.name, path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    return found


def linked(meta):
    """Package ids reachable from the root crate through normal dependency edges."""
    nodes = {node["id"]: node for node in meta["resolve"]["nodes"]}
    seen, queue = set(), [meta["resolve"]["root"]]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        for dep in nodes[current]["deps"]:
            # A missing `dep_kinds` predates cargo reporting them; treat it as normal rather
            # than dropping the crate from a notice that has to be complete.
            kinds = dep.get("dep_kinds")
            if not kinds or any(kind.get("kind") is None for kind in kinds):
                queue.append(dep["pkg"])
    return seen


def main(manifest):
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--manifest-path", manifest],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )

    reachable = linked(meta)
    crates, bodies, missing = [], {}, []
    for pkg in sorted(meta["packages"], key=lambda p: (p["name"], p["version"])):
        if pkg["name"] in OURS or pkg["id"] not in reachable:
            continue
        crates.append(pkg)
        found = texts(Path(pkg["manifest_path"]).parent)
        if not found:
            missing.append(pkg)
        for filename, body in found:
            bodies.setdefault(hashlib.sha256(body.encode()).hexdigest(), (body, filename, []))[2].append(pkg["name"])

    out = [
        "# Third-party licenses",
        "",
        "`libeqty_lineage_nemo_relay` statically links the crates below. Their licenses and",
        "notices are reproduced here; each applies to that crate's code, not to this project.",
        "",
        "Normal dependencies only -- test and build-time crates do not reach the library.",
        "",
        f"{len(crates)} crates, {len(bodies)} distinct notices.",
        "",
        "## Crates",
        "",
        "| Crate | Version | License | Source |",
        "| --- | --- | --- | --- |",
    ]
    for pkg in crates:
        out.append(
            f"| {pkg['name']} | {pkg['version']} | {pkg.get('license') or '(not declared)'} "
            f"| {pkg.get('repository') or ''} |"
        )

    if missing:
        out += [
            "",
            "## Crates shipping no license file",
            "",
            "Their declared terms are in the table above; the text was not in the published crate.",
            "",
        ]
        out += [f"- {p['name']} {p['version']} — {p.get('license') or '(not declared)'}" for p in missing]

    out += ["", "## Notices", ""]
    for _, (body, filename, names) in sorted(bodies.items(), key=lambda kv: kv[1][2][0]):
        shared = ", ".join(sorted(set(names)))
        out += [f"### {filename} — {shared}", "", "```", body.rstrip(), "```", ""]

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
