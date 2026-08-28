"""Cases where the handler could attest something the run did not actually do.

Every test here guards a way the recorded filesystem can drift from the real one. Drift in this direction
is worse than a gap: a manifest that omits a write is incomplete, but one that records a version the file
never held, or splits a file in two, is wrong in a way a reader cannot detect.
"""

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


def test_normalization_matches_the_backend():
    assert _normalize_path("report.md") == "/report.md"
    # the naive "/" + path yields "//notes.md": POSIX gives two leading slashes a meaning of their own
    assert _normalize_path("/notes.md") == "/notes.md"
    assert _normalize_path("/./foo//bar") == "/foo/bar"


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


def test_a_partly_unreadable_skill_catalogue_is_still_claimed(recording_handler, monkeypatch):
    """Declining halfway would link the skills seen so far *and* re-embed the whole catalogue."""
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
