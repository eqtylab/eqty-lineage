"""The virtual filesystem is a set of versioned entities, not a blob repeated in every node's state.

A deep agent carries its whole filesystem in graph state, so every node's state asset would otherwise
embed every file, once per node, and no file would ever be an entity of its own. Worse, a file written by
a tool would be an input to everything that read it afterwards and an output of nothing -- which in a
provenance graph says the run found it already there.
"""

from langchain_core.messages import AIMessage, HumanMessage

from scripted import call, deep_agent

FIRST = "draft one\n"
SECOND = "draft two, revised\n"


def _run(handler, script, state=None):
    return deep_agent(script).invoke(
        {"messages": [HumanMessage("write the report")], **(state or {})},
        config={"callbacks": [handler], "recursion_limit": 60},
    )


def test_each_file_is_its_own_asset(recording_handler):
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/notes.md", content="research notes\n"),
            call("write_file", "b", file_path="/report.md", content=FIRST),
            AIMessage(content="done"),
        ],
    )

    paths = {path for path, _ in recording_handler._file_versions}
    assert paths == {"/notes.md", "/report.md"}
    assert len(recording_handler._file_versions) == 2, "one version each, not one per node that saw them"


def test_a_written_file_is_an_output_of_the_call_that_wrote_it(recording_handler):
    _run(recording_handler, [call("write_file", "a", file_path="/report.md", content=FIRST), AIMessage("done")])

    written = str(recording_handler._file_versions[("/report.md", _digest(FIRST))])
    assert written in recording_handler.outputs_of("write_file"), (
        "a file that is an output of nothing reads as data the run was given rather than data it produced"
    )


def test_a_rewrite_is_a_new_version_chained_to_the_one_it_replaced(recording_handler):
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/report.md", content=FIRST),
            call("write_file", "b", file_path="/report.md", content=SECOND),
            AIMessage(content="done"),
        ],
    )

    first = str(recording_handler._file_versions[("/report.md", _digest(FIRST))])
    second = str(recording_handler._file_versions[("/report.md", _digest(SECOND))])
    assert first != second

    second_write = [c for c in recording_handler.computations if c[0] == "write_file"][1]
    assert first in second_write[2], "the version being replaced is what the rewrite was made against"
    assert second in second_write[3], "the new version is what the rewrite produced"


def test_an_edit_is_attributed_to_the_call_that_made_it(recording_handler):
    """``edit_file`` reports only that it succeeded, so the result has to be derived from the version it
    was made against -- and it must be derived exactly, or the chain records content the run never had."""
    result = _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/report.md", content=FIRST),
            call("edit_file", "b", file_path="/report.md", old_string="draft one", new_string="draft two"),
            AIMessage(content="done"),
        ],
    )

    actual = result["files"]["/report.md"]["content"]
    assert actual == "draft two\n"

    edited = str(recording_handler._file_versions[("/report.md", _digest(actual))])
    assert edited in recording_handler.outputs_of("edit_file")

    original = str(recording_handler._file_versions[("/report.md", _digest(FIRST))])
    assert original in recording_handler.inputs_of("edit_file")


def test_a_failed_edit_records_no_version(recording_handler):
    """The tool reports a missed match in its result rather than by raising, and a version invented for a
    write that never happened is worse than no version at all."""
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/report.md", content=FIRST),
            call("edit_file", "b", file_path="/report.md", old_string="not in the file", new_string="x"),
            AIMessage(content="done"),
        ],
    )

    assert len(recording_handler._file_versions) == 1, "a failed edit must not mint a version"


def test_a_file_read_is_an_input_to_the_call_that_read_it(recording_handler):
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/report.md", content=FIRST),
            call("read_file", "b", file_path="/report.md"),
            AIMessage(content="done"),
        ],
    )

    version = str(recording_handler._file_versions[("/report.md", _digest(FIRST))])
    assert version in recording_handler.inputs_of("read_file")


def test_unchanged_content_is_carried_rather_than_re_registered(recording_handler):
    """The filesystem is in the state of every model turn; a file must not become a new asset each time."""
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/report.md", content=FIRST),
            call("read_file", "b", file_path="/report.md"),
            call("read_file", "c", file_path="/report.md"),
            AIMessage(content="done"),
        ],
    )

    assert len(recording_handler._file_versions) == 1
    version = str(recording_handler._file_versions[("/report.md", _digest(FIRST))])
    produced = [name for name, _, _, outs in recording_handler.computations if version in outs]
    assert produced == ["write_file"], f"the version was re-emitted as an output by {produced}"


def test_the_filesystem_is_not_embedded_in_every_state_asset(recording_handler, monkeypatch):
    """The whole point of the extractor: state payloads name the files, they do not contain them."""
    import eqty_lineage.langchain as handler_module

    payloads = []
    original = handler_module.Dataset.from_object

    def record(obj, *args, **kwargs):
        payloads.append(obj)
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(handler_module.Dataset, "from_object", record)

    marker = "a sentence that appears in exactly one file\n"
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/report.md", content=marker),
            call("read_file", "b", file_path="/report.md"),
            AIMessage(content="done"),
        ],
    )

    embedded = [p for p in payloads if marker in _flatten(p)]
    # the tool's arguments and the tool's result legitimately hold the content; every *state* payload
    # should hold the file's CID instead
    assert len(embedded) <= 2, f"the file's content was embedded in {len(embedded)} state payloads"


def _flatten(payload) -> str:
    import json

    return json.dumps(payload, default=str)


def _digest(content: str) -> str:
    from eqty_lineage.deepagents import _digest as digest

    return digest(content)


# A file asset's CID has to *be* its content's CID. Wrapping the payload as
# `{"path": ..., "content": ...}` made it the hash of a JSON envelope instead, which broke three
# things at once: two identical files at different paths got different CIDs, the digest depended on
# `json.dumps` key order and separator whitespace, and Lineage Explorer rendered a JSON blob where the
# file was supposed to be. Nothing pinned the property before, so the regression shipped in 0.1.0.
def test_file_asset_cid_is_the_content_cid(recording_handler):
    """The stored blob is the file, byte for byte -- not a JSON object describing it."""
    from eqty_sdk import get_cid_for_bytes

    content = "# Report\n\nFindings.\n\n- one\n- two\n"
    cid, created, _ = recording_handler.register_virtual_file("/report.md", content, {})

    assert created, "the version should have been registered"
    assert str(cid) == str(get_cid_for_bytes(content.encode("utf-8"), False)), (
        "the Document's CID is not the CID of its content, so the payload is not the file itself"
    )


def test_identical_content_at_two_paths_is_one_asset(recording_handler):
    """Content addressing means the bytes decide identity; the path is metadata.

    This is the property the JSON envelope destroyed, and it is also the thing to watch when changing
    the payload: `_file_versions` stays keyed per path, so both paths still get their own version
    record and their own lineage, while the asset they point at is shared.
    """
    content = "identical\n"
    first, first_created, _ = recording_handler.register_virtual_file("/a.md", content, {})
    second, _, _ = recording_handler.register_virtual_file("/b.md", content, {})

    assert first_created
    assert str(first) == str(second), "same bytes at two paths should be one content-addressed asset"
    assert recording_handler._file_latest["/a.md"] == recording_handler._file_latest["/b.md"]
