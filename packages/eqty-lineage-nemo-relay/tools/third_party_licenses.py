"""Collect available notices from the plugin's target-filtered normal dependencies.

Cargo's dependency graph is a conservative inventory, not a binary contents audit.
Nested notices may cover vendored code; identical texts are emitted once.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

LICENSE_FILES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "UNLICENSE")
OURS = {"eqty-lineage-nemo-relay", "eqty-lineage-nemo-relay-abi-test"}


def texts(crate_dir, license_file=None):
    """Include declared license paths as well as conventional nested notices."""
    crate_dir = crate_dir.resolve(strict=True)
    paths = set()
    if license_file:
        declared = Path(license_file)
        if not declared.is_absolute():
            declared = crate_dir / declared
        # A published crate need not ship the path its manifest declares -- `license-file` is often
        # a repository path excluded from the tarball. The walk below still finds a conventional
        # notice, and a crate with neither is reported as missing, so the packaging run should not
        # fail over one manifest field. Strictness is kept where a silent gap would matter: the walk
        # names files it just listed, and a text that cannot be read there is a real failure.
        try:
            paths.add(declared.resolve(strict=True))
        except OSError:
            pass

    def scan_error(error):
        raise error

    for directory, dirs, files in os.walk(crate_dir, onerror=scan_error):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "target"})
        for name in files:
            if name.upper().startswith(LICENSE_FILES):
                paths.add((Path(directory) / name).resolve(strict=True))
    found = []
    for path in sorted(paths):
        name = os.path.relpath(path, crate_dir)
        # Replace rather than raise. A notice is reproduced for a reader, not parsed, and a latin-1
        # copyright line is common enough that failing on it would block a release over a byte in
        # someone's name. The walk found the file, so the notice is present either way.
        found.append((name, path.read_text(encoding="utf-8", errors="replace")))
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


def main(manifest, target):
    meta = json.loads(
        subprocess.run(
            [
                "cargo",
                "metadata",
                "--locked",
                "--offline",
                "--format-version",
                "1",
                "--filter-platform",
                target,
                "--manifest-path",
                manifest,
            ],
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
        found = texts(Path(pkg["manifest_path"]).parent, pkg.get("license_file"))
        if not found:
            missing.append(pkg)
        for filename, body in found:
            bodies.setdefault(hashlib.sha256(body.encode()).hexdigest(), (body, filename, []))[2].append(
                f"{pkg['name']} {pkg['version']} ({filename})"
            )

    out = [
        "# Third-party licenses",
        "",
        "Available license texts and notices from the plugin's normal dependency graph.",
        "This is a conservative source inventory, not an exact inventory of linked code.",
        "",
        f"Build target: `{target}`. Test and build dependency edges are excluded.",
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
            "## Crates with no collected license text",
            "",
            "No conventional notice or declared license file was found in these crate sources.",
            "Their declared terms are listed above; this report does not resolve missing texts.",
            "",
        ]
        out += [f"- {p['name']} {p['version']} — {p.get('license') or '(not declared)'}" for p in missing]

    out += ["", "## Notices", ""]
    for _, (body, filename, names) in sorted(bodies.items(), key=lambda kv: kv[1][2][0]):
        shared = ", ".join(sorted(set(names)))
        out += [f"### {filename} — {shared}", "", "```", body.rstrip(), "```", ""]

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--target", required=True, help="The target triple used to build the plugin")
    args = parser.parse_args()
    main(args.manifest, args.target)
