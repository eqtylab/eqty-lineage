"""A file rewritten mid-run is a new entity, not the one already registered.

Keying registered paths on the path alone made the second sighting of a rewritten file resolve to the
first sighting's asset. Everything downstream was then linked to content it never saw -- an attestation
that is not merely incomplete but wrong.
"""

from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class DocState(TypedDict):
    trail: Annotated[list, lambda a, b: a + b]
    report: Path


def _rewrite_graph(target: Path, second: str):
    """author (sees v1) -> revise (writes v2) -> publish (must see v2)."""

    def author(state: DocState) -> dict:
        return {"trail": ["author"], "report": target}

    def revise(state: DocState) -> dict:
        target.write_text(second)
        return {"trail": ["revise"], "report": target}

    def publish(state: DocState) -> dict:
        return {"trail": ["publish"], "report": target}

    graph = StateGraph(DocState)
    for name, fn in (("author", author), ("revise", revise), ("publish", publish)):
        graph.add_node(name, fn)
    graph.add_edge(START, "author")
    graph.add_edge("author", "revise")
    graph.add_edge("revise", "publish")
    graph.add_edge("publish", END)
    return graph.compile()


def test_rewrite_creates_a_second_version(recording_handler, tmp_path):
    report = tmp_path / "report.md"
    report.write_text("version one\n")

    _rewrite_graph(report, "version two, revised\n").invoke(
        {"trail": [], "report": report}, config={"callbacks": [recording_handler]}
    )

    versions = list(recording_handler._path_versions.values())
    assert len(versions) == 2, f"expected two versions of one path, got {len(versions)}"

    paths = {path for path, _ in recording_handler._path_versions}
    assert paths == {str(report.resolve())}, "both versions should belong to the same path"


def test_rewrite_is_chained_to_what_it_replaced(recording_handler, tmp_path):
    report = tmp_path / "report.md"
    report.write_text("version one\n")

    _rewrite_graph(report, "version two, revised\n").invoke(
        {"trail": [], "report": report}, config={"callbacks": [recording_handler]}
    )

    first, second = list(recording_handler._path_versions.values())
    revise_in = recording_handler.inputs_of("revise")
    revise_out = recording_handler.outputs_of("revise")

    assert str(first) in revise_in, "the replaced version should be an input to the node that rewrote it"
    assert str(second) in revise_out, "the new version should be an output of the node that wrote it"


def test_downstream_reads_the_current_version(recording_handler, tmp_path):
    """The regression: 'publish' used to be linked to the pre-rewrite asset."""
    report = tmp_path / "report.md"
    report.write_text("version one\n")

    _rewrite_graph(report, "version two, revised\n").invoke(
        {"trail": [], "report": report}, config={"callbacks": [recording_handler]}
    )

    first, second = list(recording_handler._path_versions.values())
    publish_in = recording_handler.inputs_of("publish")

    assert str(second) in publish_in, "publish must be linked to the content it actually read"
    assert str(first) not in publish_in, "publish must not be linked to the superseded content"


def test_unchanged_file_is_carried_not_recreated(recording_handler, tmp_path):
    """Identical bytes at the same path stay one entity across every node that sees them."""
    report = tmp_path / "steady.md"
    report.write_text("unchanged\n")

    # `second` is the same content, so no new version should appear
    _rewrite_graph(report, "unchanged\n").invoke(
        {"trail": [], "report": report}, config={"callbacks": [recording_handler]}
    )

    assert len(recording_handler._path_versions) == 1
    only = str(next(iter(recording_handler._path_versions.values())))
    # carried through as an input, never re-emitted as an output of a later node
    assert only not in recording_handler.outputs_of("publish")


def test_missing_path_does_not_break_the_run(recording_handler, tmp_path):
    """An unreadable path is skipped, not fatal -- the handler must not take down the run it observes."""
    missing = tmp_path / "never-created.md"

    def only_node(state: DocState) -> dict:
        return {"trail": ["only"], "report": missing}

    graph = StateGraph(DocState)
    graph.add_node("only", only_node)
    graph.add_edge(START, "only")
    graph.add_edge("only", END)

    result = graph.compile().invoke({"trail": [], "report": missing}, config={"callbacks": [recording_handler]})
    assert result["trail"] == ["only"]
    assert recording_handler._path_versions == {}
