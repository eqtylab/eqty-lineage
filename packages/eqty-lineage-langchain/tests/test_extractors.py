"""The extension point a framework package uses to teach the handler about its own state.

Without it, everything a node's state holds is re-serialized into that node's state Dataset every time,
so a filesystem carried in state is embedded once per node and no file is ever an entity in its own
right. These tests fix the contract the DeepAgents package will depend on.
"""

import json
from typing import Annotated, Any, TypedDict

from eqty_lineage.langchain import UNCLAIMED, AssetSink, StateExtractor
from eqty_sdk import Dataset
from langgraph.graph import END, START, StateGraph


class S(TypedDict):
    trail: Annotated[list, lambda a, b: a + b]
    payload: Any


def _graph(fn):
    graph = StateGraph(S)
    graph.add_node("work", fn)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


class FilesExtractor(StateExtractor):
    """Claims a `files` mapping, exactly as the DeepAgents package will."""

    def __init__(self) -> None:
        self.seen: list[tuple] = []

    def extract(self, key_path, value, sink: AssetSink):
        if key_path != ("payload",) or not isinstance(value, dict):
            return UNCLAIMED
        self.seen.append((key_path, tuple(sorted(value))))
        for path, content in sorted(value.items()):
            asset = Dataset.from_object({"path": path, "content": content}, name=path, **sink.metadata)
            sink.create(asset.cid)
        return {"extracted": sorted(value)}


def test_extractor_claims_its_key(recording_handler):
    extractor = FilesExtractor()
    recording_handler.add_extractor(extractor)

    _graph(lambda s: {"trail": ["x"], "payload": {"/a.md": "one", "/b.md": "two"}}).invoke(
        {"trail": [], "payload": {}}, config={"callbacks": [recording_handler]}
    )

    assert extractor.seen, "the extractor was never consulted for its key"
    assert ("/a.md", "/b.md") in [keys for _, keys in extractor.seen]


def test_claimed_content_is_not_inlined_in_the_state_blob(recording_handler, monkeypatch):
    """The point of D5: what became its own asset must not also be re-embedded in every node's state.

    Asserted against the payload actually handed to the SDK, rather than by re-rendering it through the
    same extractor -- which would claim the content a second time and prove nothing.
    """
    import eqty_lineage.langchain as mod

    recorded: list[tuple] = []
    real = mod.Dataset._from_object

    def spy(obj, asset_type, ctx=None, _store=None, **kwargs):
        recorded.append((kwargs.get("name"), obj))
        return real(obj, asset_type, ctx, _store, **kwargs)

    # patched below the public `from_object` classmethod: `_asset_factory` binds a context and calls
    # `_from_object` directly, skipping `from_object` entirely, so that's the only point both the
    # extractor's direct `Dataset.from_object` and the handler's context-bound calls both pass through.
    monkeypatch.setattr(mod.Dataset, "_from_object", classmethod(lambda cls, *a, **kw: spy(*a, **kw)))
    recording_handler.add_extractor(FilesExtractor())

    _graph(lambda s: {"trail": ["x"], "payload": {"/a.md": "SECRET-CONTENT"}}).invoke(
        {"trail": [], "payload": {}}, config={"callbacks": [recording_handler]}
    )

    state_payloads = [(name, p) for name, p in recorded if "state" in (name or "")]
    assert state_payloads, "no state assets were registered"
    for name, payload in state_payloads:
        assert "SECRET-CONTENT" not in json.dumps(payload), f"claimed content still inlined in '{name}'"

    # and it exists exactly where it belongs -- the asset the extractor made for it
    own = [p for name, p in recorded if name == "/a.md"]
    assert own and own[0]["content"] == "SECRET-CONTENT"


def test_registration_order_puts_the_newest_first(recording_handler):
    """A subclass's extractor must beat the built-in PathExtractor."""

    class Greedy(StateExtractor):
        def extract(self, key_path, value, sink):
            return "claimed" if key_path == ("payload",) else UNCLAIMED

    recording_handler.add_extractor(Greedy())
    assert isinstance(recording_handler._extractors[0], Greedy)


def test_a_broken_extractor_does_not_break_the_run(recording_handler):
    class Broken(StateExtractor):
        def extract(self, key_path, value, sink):
            raise RuntimeError("extractor is broken")

    recording_handler.add_extractor(Broken())

    result = _graph(lambda s: {"trail": ["survived"], "payload": {"a": 1}}).invoke(
        {"trail": [], "payload": {}}, config={"callbacks": [recording_handler]}
    )
    assert result["trail"] == ["survived"]
    assert recording_handler.computations, "the run should still be recorded"


def test_path_extractor_is_registered_by_default(recording_handler):
    from eqty_lineage.langchain import PathExtractor

    assert any(isinstance(e, PathExtractor) for e in recording_handler._extractors)
