"""Cases where the handler could attest something the run did not actually do.

Every test here guards a way the recorded filesystem can drift from the real one. Drift in this direction
is worse than a gap: a manifest that omits a write is incomplete, but one that records a version the file
never held, or splits a file in two, is wrong in a way a reader cannot detect.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from scripted import call, deep_agent

from eqty_lineage.deepagents import _digest, _is_tool_error, _normalize_path
from eqty_lineage.deepagents.extractors import _file_content


def _run(handler, script, state=None):
    return deep_agent(script).invoke(
        {"messages": [HumanMessage("go")], **(state or {})},
        config={"callbacks": [handler], "recursion_limit": 60},
    )


def test_an_un_normalized_path_is_the_same_file(recording_handler):
    """DeepAgents runs every path through `validate_path`, so `report.md` becomes `/report.md` in state.

    Keying the raw argument made the write land on one entity and every later read on another: the file
    the agent produced ended up an output of nothing, which reads as data the run was given. Live models
    omit the leading slash routinely.
    """
    result = _run(
        recording_handler,
        [
            call("write_file", "a", file_path="report.md", content="draft one\n"),
            call("read_file", "b", file_path="report.md"),
            AIMessage(content="done"),
        ],
    )

    assert sorted(result["files"]) == ["/report.md"]
    assert {path for path, _ in recording_handler._file_versions} == {"/report.md"}

    version = str(next(iter(recording_handler._file_versions.values())))
    assert version in recording_handler.outputs_of("write_file")
    assert version in recording_handler.inputs_of("read_file")


#: paths the backend normalizes, and paths it refuses. Kept together so one test can assert that this
#: package agrees with DeepAgents on both -- which of the two happens is itself part of the agreement.
NORMALIZATION_CASES = [
    "report.md",
    "/report.md",
    # POSIX gives a doubled leading slash a meaning of its own and `normpath` preserves exactly two, so
    # this is a *different file* from /report.md. Collapsing it merged one file with its neighbour, and a
    # write to one was then attested as a rewrite of the other.
    "//report.md",
    "///report.md",
    "/./foo//bar",
    "//a/b",
    "/a/",
    "/",
    ".",
    "a/b/c.txt",
    "/dir/../file.md",
    "../etc/passwd",
    "~/secrets",
    "C:/Users/file.txt",
    "d:\\data\\file.txt",
    ".\\x",
    "/a\\b",
]


@pytest.mark.parametrize("path", NORMALIZATION_CASES)
def test_normalization_matches_the_backend(path):
    """`_normalize_path` must agree with DeepAgents' `validate_path` exactly, not approximately.

    Near agreement is the worst outcome available: two paths the backend keeps apart but this folds
    together are two real files recorded as one asset. Asserted against the real function rather than
    against a table of expected strings, so the day upstream changes its rules this fails instead of
    drifting silently. `deepagents` is imported here and nowhere in the shipped package.
    """
    from deepagents.backends.utils import validate_path

    try:
        expected = validate_path(path)
    except ValueError:
        assert _normalize_path(path) is None, f"'{path}' is refused by the backend and must not be keyed"
    else:
        assert _normalize_path(path) == expected


#: (content, old_string, new_string, replace_all) -- edits the backend applies, and edits it refuses.
#: Which of the two happens is part of the agreement, so both kinds are in one table.
EDIT_CASES = [
    ("draft one\n", "draft one", "draft two", False),
    # the same string twice: refused without replace_all, applied to both with it
    ("a\na\n", "a", "b", False),
    ("a\na\n", "a", "b", True),
    ("a\na\na\n", "a", "b", True),
    # not present at all
    ("hello\n", "goodbye", "x", False),
    # the EOF-newline case the backend detects specially: old_string carries a trailing newline the
    # file lacks at that position. It errors today, but it is one upstream change from succeeding.
    ("hello", "hello\n", "x\n", False),
    ("one\ntwo", "two\n", "three\n", False),
    # deletion, and a no-op replacement
    ("keep\nthis\n", "this\n", "", False),
    ("same\n", "same", "same", False),
    # a substring that also occurs inside a longer word
    ("cat catalog\n", "cat", "dog", True),
    ("cat catalog\n", "cat", "dog", False),
    # multi-line old_string
    ("a\nb\nc\n", "a\nb", "z", False),
    ("", "x", "y", False),
]


@pytest.mark.parametrize(("content", "old", "new", "replace_all"), EDIT_CASES)
def test_edit_reconstruction_matches_the_backend(recording_handler, content, old, new, replace_all):
    """`_apply_edit` must agree with DeepAgents' `perform_string_replacement` exactly.

    `edit_file` reports only that it succeeded, so the resulting content is derived from the version the
    edit was made against -- which means this package carries a copy of the backend's occurrence rules.
    A copy that drifts is worse than no copy: too permissive and it mints a version the file never held;
    too strict and the edit silently records nothing. Asserted against the real function, so upstream
    changing its rules fails here rather than going unnoticed. `deepagents` is imported in the tests and
    nowhere in the shipped package.
    """
    from deepagents.backends.utils import perform_string_replacement

    path = "/subject.md"
    recording_handler._file_contents[path] = content
    pending = {"path": path, "old_string": old, "new_string": new, "replace_all": replace_all}

    result = perform_string_replacement(content, old, new, replace_all=replace_all)
    reconstructed = recording_handler._apply_edit(path, pending)

    if isinstance(result, str):
        # the backend refused the edit; nothing was written, so nothing may be registered
        assert reconstructed is None, f"backend refused ({result[:40]}...) but a version was reconstructed"
    else:
        expected, _occurrences = result
        assert reconstructed == expected


def test_an_edit_against_unseen_content_reconstructs_nothing(recording_handler):
    """Without the version the edit was made against there is nothing to derive the result from, and a
    guess would be a file version the run never had."""
    pending = {"path": "/never-seen.md", "old_string": "a", "new_string": "b", "replace_all": False}
    assert recording_handler._apply_edit("/never-seen.md", pending) is None


def test_a_refused_path_records_nothing(recording_handler):
    """A path the backend refuses never reaches it, so the call touched no file."""
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="../outside.md", content="nope\n"),
            AIMessage(content="done"),
        ],
    )

    assert recording_handler._file_versions == {}
    assert recording_handler._file_latest == {}


def test_a_backend_failure_that_does_not_say_error_registers_nothing(recording_handler):
    """Only some backends word a failure "Error". The store, LangSmith and sandbox backends report
    "Failed to write file ..." or the remote's own message, and a prefix match takes those for successes."""
    assert _is_tool_error(ToolMessage(content="Hub unavailable: boom", tool_call_id="x", status="error"))
    assert _is_tool_error(ToolMessage(content="Error: string not found", tool_call_id="x"))
    assert not _is_tool_error(ToolMessage(content="Updated file /report.md", tool_call_id="x"))


def test_a_revert_is_recorded_as_a_write(recording_handler):
    """Restoring earlier bytes mints no new asset, but it is still a write.

    Skipping it left the manifest asserting that the version it replaced was still current -- the file
    ends the run holding A with nothing after the write of B having produced it.
    """
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/r.md", content="A\n"),
            call("write_file", "b", file_path="/r.md", content="B\n"),
            call("write_file", "c", file_path="/r.md", content="A\n"),
            AIMessage(content="done"),
        ],
    )

    version_a = str(recording_handler._file_versions[("/r.md", _digest("A\n"))])
    version_b = str(recording_handler._file_versions[("/r.md", _digest("B\n"))])

    writes = [c for c in recording_handler.computations if c[0] == "write_file"]
    assert len(writes) == 3
    assert version_a in writes[2][3], "the restored version is what the third write produced"
    assert version_b in writes[2][2], "and it replaced the version that was current"


def test_a_delete_stops_the_file_being_current(recording_handler):
    """A deleted file kept as `latest` chains a later write to content that no longer existed."""
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/r.md", content="A\n"),
            call("delete", "b", file_path="/r.md"),
            call("write_file", "c", file_path="/r.md", content="C\n"),
            AIMessage(content="done"),
        ],
    )

    version_a = str(recording_handler._file_versions[("/r.md", _digest("A\n"))])
    assert version_a in recording_handler.inputs_of("delete"), "the delete acted on the live version"

    second_write = [c for c in recording_handler.computations if c[0] == "write_file"][1]
    assert version_a not in second_write[2], "a write after a delete replaces nothing"


def test_legacy_list_content_is_read_not_skipped():
    """A checkpoint written by an older DeepAgents stores content as a list of lines, which
    `file_data_to_string` still joins. Reading only `str` would drop a resumed run's whole filesystem."""
    assert _file_content({"content": ["line one", "line two"]}) == "line one\nline two"
    assert _file_content({"content": "plain\n"}) == "plain\n"
    assert _file_content({"content": 42}) is None


def test_a_written_file_is_never_both_input_and_output(recording_handler):
    """The handler hands a written file to the finalize that follows; if the tool's result also carries
    it, the extractor has already linked it as an input, and adding it again makes a self-cycle."""
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/r.md", content="A\n"),
            call("edit_file", "b", file_path="/r.md", old_string="A", new_string="B"),
            AIMessage(content="done"),
        ],
    )

    for name, _kind, inputs, outputs in recording_handler.computations:
        assert not (set(inputs) & set(outputs)), f"'{name}' is its own ancestor"


def test_a_partly_unreadable_skill_catalogue_is_declined_whole(recording_handler):
    """An unreadable entry anywhere makes the extractor decline the catalogue rather than claim part of it.

    Declining costs the state blob the catalogue either way. What declining *upfront* avoids is doing that
    while also having linked the skills seen so far as inputs -- a manifest claiming the turn was given
    three skills when the state held five is worse than one that says nothing.
    """
    from eqty_lineage.deepagents.extractors import SkillExtractor
    from eqty_lineage.langchain import UNCLAIMED, AssetSink

    sink = AssetSink({})
    extractor = SkillExtractor(recording_handler)
    claimed = extractor.extract(("skills_metadata",), [{"name": "ok", "description": "d"}, "legacy"], sink)

    assert claimed is UNCLAIMED
    assert sink.carried == [], "nothing may be registered when the catalogue is not fully readable"


def test_a_credential_in_the_system_prompt_is_redacted(recording_handler, monkeypatch):
    """A middleware is free to interpolate configuration into the prompt it assembles, and the prompt is
    stored as a content-addressed blob on disk -- so a leak here is a credential written out and CID'd."""
    import eqty_lineage.deepagents as handler_module
    from langchain_core.messages import SystemMessage

    payloads = []
    original = handler_module.SystemPrompt.from_object

    def record(obj, *args, **kwargs):
        payloads.append(obj)
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(handler_module.SystemPrompt, "from_object", record)

    secret = "sk-live-should-not-appear"
    prompt = SystemMessage(content=[{"type": "text", "text": "You are an agent."}, {"api_key": secret}])
    recording_handler._register_system_prompt([[prompt]], "model")

    assert payloads, "the prompt should have been registered"
    assert secret not in str(payloads[0])
    assert "You are an agent." in str(payloads[0]), "only the credential is removed"


# ------------------------------------------------------------------ files the run only reads ----
#
# A filesystem, store or sandbox backend keeps the virtual filesystem out of graph state entirely, so
# neither the extractor nor the write path ever sees a file that already existed and is never written --
# which is every source file such an agent reads. These cover that case, and the ways recording it could
# go wrong.


def _fs_agent(script, root):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    from scripted import ScriptedModel

    return create_deep_agent(
        model=ScriptedModel(messages=iter(script)),
        backend=FilesystemBackend(root_dir=str(root)),
        name="fs-agent",
    )


def _fs_run(handler, script, root):
    return _fs_agent(script, root).invoke(
        {"messages": [HumanMessage("go")]},
        config={"callbacks": [handler], "recursion_limit": 60},
    )


def test_a_file_only_ever_read_still_becomes_an_entity(recording_handler, tmp_path):
    """Without this the model's answer derives from a `read_file` that consumed nothing, and the file it
    was actually built from is absent from the graph."""
    (tmp_path / "seed.md").write_text("pre-existing content\n")

    result = _fs_run(
        recording_handler,
        [call("read_file", "a", file_path="/seed.md"), AIMessage(content="done")],
        tmp_path,
    )

    assert "files" not in result, "the filesystem backend keeps files off the state"
    assert len(recording_handler._read_cids) == 1
    (path, _digest_of_rendering), cid = next(iter(recording_handler._read_cids.items()))
    assert path == "/seed.md"
    assert str(cid) in recording_handler.inputs_of("read_file"), "the run consumed it"
    assert str(cid) not in recording_handler.outputs_of("read_file"), "the run did not produce it"


def test_a_read_rendering_never_stands_in_for_a_file_the_run_wrote(recording_handler, tmp_path):
    """A file the run wrote has exact bytes; reading it back must link those, not how it was rendered."""
    _fs_run(
        recording_handler,
        [
            call("write_file", "a", file_path="/r.md", content="exact bytes\n"),
            call("read_file", "b", file_path="/r.md"),
            AIMessage(content="done"),
        ],
        tmp_path,
    )

    assert recording_handler._read_cids == {}, "no rendering asset for a path with real bytes"
    written = str(recording_handler._file_versions[("/r.md", _digest("exact bytes\n"))])
    assert written in recording_handler.inputs_of("read_file")


def test_a_rendering_is_never_edited_against(recording_handler, tmp_path):
    """The safety property. The tool's result is line-numbered and drops a trailing newline, so it is
    lossy in a way that cannot be undone -- letting it become the version an `edit_file` reconstructs
    against would mint content the file never held."""
    (tmp_path / "seed.md").write_text("alpha\nbeta\n")

    _fs_run(
        recording_handler,
        [
            call("read_file", "a", file_path="/seed.md"),
            call("edit_file", "b", file_path="/seed.md", old_string="alpha", new_string="omega"),
            AIMessage(content="done"),
        ],
        tmp_path,
    )

    assert "/seed.md" not in recording_handler._file_contents, "a rendering must not be cached as content"
    assert recording_handler._file_versions == {}, "the edit had no exact base, so it minted no version"
    # the edit really did happen on disk -- the handler declined to guess at it, rather than missing it
    assert (tmp_path / "seed.md").read_text() == "omega\nbeta\n"


def test_the_same_file_read_twice_is_one_entity(recording_handler, tmp_path):
    (tmp_path / "seed.md").write_text("stable\n")

    _fs_run(
        recording_handler,
        [
            call("read_file", "a", file_path="/seed.md"),
            call("read_file", "b", file_path="/seed.md"),
            AIMessage(content="done"),
        ],
        tmp_path,
    )

    assert len(recording_handler._read_cids) == 1


def test_a_failed_read_registers_nothing(recording_handler, tmp_path):
    _fs_run(
        recording_handler,
        [call("read_file", "a", file_path="/missing.md"), AIMessage(content="done")],
        tmp_path,
    )

    assert recording_handler._read_cids == {}


# ------------------------------------------------------------------ deleting a directory ----
#
# `delete` removes a whole subtree, not one key: the backend drops `key == base` and everything under
# `base + "/"`. A handler that forgets only the exact path keeps every nested file as current, which is
# the same drift `test_a_delete_stops_the_file_being_current` guards for a single file.


def test_deleting_a_directory_forgets_every_file_under_it(recording_handler):
    """Asserted against the state the backend actually produced, so the subtree rule cannot drift.

    `/dirx.md` is the case a prefix match gets wrong: it starts with `/dir` but is not under it.
    """
    result = _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/dir/a.md", content="A\n"),
            call("write_file", "b", file_path="/dir/nested/b.md", content="B\n"),
            call("write_file", "c", file_path="/dirx.md", content="X\n"),
            call("delete", "d", file_path="/dir"),
            AIMessage(content="done"),
        ],
    )

    surviving = set(result["files"])
    assert surviving == {"/dirx.md"}, "the backend removed the subtree and kept the sibling"
    assert set(recording_handler._file_latest) == surviving
    assert set(recording_handler._file_contents) == surviving


def test_deleting_a_directory_leaves_a_sibling_with_a_shared_prefix_alone(recording_handler):
    """`/dirx.md` starts with `/dir` but is not under it, so the prefix needs its trailing slash.

    Driven straight at `_record_write` rather than through a run: under the state backend the extractor
    re-registers whatever survived in state on the very next node, which heals a too-greedy delete before
    anything can observe it. Under a filesystem or sandbox backend nothing heals it, and the sibling's
    version chain is simply lost.
    """
    from uuid import uuid4

    recording_handler._file_latest = {"/dir/a.md": "cid-a", "/dir/nested/b.md": "cid-b", "/dirx.md": "cid-x"}
    recording_handler._file_contents = {"/dir/a.md": "A\n", "/dir/nested/b.md": "B\n", "/dirx.md": "X\n"}

    recording_handler._record_write(
        uuid4(),
        {"path": "/dir", "deleted": True},
        ToolMessage(content="Deleted directory /dir", tool_call_id="t"),
    )

    assert set(recording_handler._file_latest) == {"/dirx.md"}
    assert set(recording_handler._file_contents) == {"/dirx.md"}


def test_a_write_under_a_deleted_directory_replaces_nothing(recording_handler):
    """A nested file kept as current chains the next write to content that no longer existed."""
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/dir/a.md", content="A\n"),
            call("delete", "b", file_path="/dir"),
            call("write_file", "c", file_path="/dir/a.md", content="C\n"),
            AIMessage(content="done"),
        ],
    )

    version_a = str(recording_handler._file_versions[("/dir/a.md", _digest("A\n"))])
    second_write = [c for c in recording_handler.computations if c[0] == "write_file"][1]
    assert version_a not in second_write[2], "a write after its directory was deleted replaces nothing"


def test_an_edit_under_a_deleted_directory_reconstructs_nothing(recording_handler):
    """Reconstructing against the deleted content mints a version the file never held."""
    _run(
        recording_handler,
        [
            call("write_file", "a", file_path="/dir/a.md", content="A\n"),
            call("delete", "b", file_path="/dir"),
            AIMessage(content="done"),
        ],
    )

    pending = {"path": "/dir/a.md", "old_string": "A", "new_string": "B", "replace_all": False}
    assert recording_handler._apply_edit("/dir/a.md", pending) is None


# ------------------------------------------------------------------ reverts seen in state ----
#
# `register_virtual_file` returns `replaced` independently of `created` precisely so a revert is not
# lost. Both extractors dropped it, which is the caller mistake that docstring warns about.


def test_a_file_reverted_in_state_supersedes_the_version_it_replaced(recording_handler):
    """State showing the file back at earlier bytes mints nothing, but it is still what this node left."""
    from eqty_lineage.deepagents.extractors import VirtualFileExtractor
    from eqty_lineage.langchain import AssetSink

    version_a, _, _ = recording_handler.register_virtual_file("/r.md", "A\n", {})
    version_b, _, _ = recording_handler.register_virtual_file("/r.md", "B\n", {})

    sink = AssetSink({})
    VirtualFileExtractor(recording_handler).extract(("files",), {"/r.md": {"content": "A\n"}}, sink)

    assert version_a in sink.created, "the state shows A current, so this node produced it"
    assert version_b in sink.carried, "and A supersedes B"


def test_a_plan_reverted_in_state_supersedes_the_revision_it_replaced(recording_handler):
    """The same rule for the plan: a revert that links nothing reverses the one edge that matters."""
    from eqty_lineage.deepagents.extractors import TodoListExtractor
    from eqty_lineage.langchain import AssetSink

    plan_a = [{"content": "step one", "status": "pending"}]
    plan_b = [{"content": "step two", "status": "pending"}]
    revision_a, _, _ = recording_handler.register_todos(plan_a, {})
    revision_b, _, _ = recording_handler.register_todos(plan_b, {})

    sink = AssetSink({})
    TodoListExtractor(recording_handler).extract(("todos",), plan_a, sink)

    assert revision_a in sink.created
    assert revision_b in sink.carried


def test_a_plan_reverted_to_an_earlier_revision_is_recorded_as_a_write(recording_handler):
    """End to end: `write_todos` back to an earlier plan is a write, not a read of what it wrote."""
    _run(
        recording_handler,
        [
            call("write_todos", "a", todos=[{"content": "one", "status": "pending"}]),
            call("write_todos", "b", todos=[{"content": "two", "status": "pending"}]),
            call("write_todos", "c", todos=[{"content": "one", "status": "pending"}]),
            AIMessage(content="done"),
        ],
    )

    revision_a = str(recording_handler._todo_versions[_digest([{"content": "one", "status": "pending"}])])
    revision_b = str(recording_handler._todo_versions[_digest([{"content": "two", "status": "pending"}])])

    writes = [c for c in recording_handler.computations if c[0] == "write_todos"]
    assert len(writes) == 3
    assert revision_a in writes[2][3], "the restored revision is what the third call produced"
    assert revision_a not in writes[2][2], "not an input to the call that wrote it"
    assert revision_b in writes[2][2], "and it replaced the revision that was current"


# ------------------------------------------------------------------ skill payloads ----


def test_a_credential_in_a_skill_is_redacted(recording_handler, monkeypatch):
    """Redaction has to run on the *serialized* payload, the way the system prompt does it.

    `_redact` only walks dicts and lists, so an object it cannot see into passes through untouched --
    and `_to_jsonable` then expands it via `model_dump()` into a dict whose credential key is never
    re-examined. Skills are content-addressed blobs on disk, so that is a secret written out.
    """
    import eqty_lineage.deepagents as handler_module

    payloads = []
    original = handler_module.Skill.from_object

    def record(obj, *args, **kwargs):
        payloads.append(obj)
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(handler_module.Skill, "from_object", record)

    class _Config:
        """Anything `_redact` cannot walk but `_to_jsonable` can expand."""

        def model_dump(self):
            return {"api_key": "sk-live-should-not-appear", "endpoint": "https://example.test"}

    recording_handler.register_skill({"name": "deploy", "description": "d", "config": _Config()}, {})

    assert payloads, "the skill should have been registered"
    assert "sk-live-should-not-appear" not in str(payloads[0])
    assert "https://example.test" in str(payloads[0]), "only the credential is removed"
