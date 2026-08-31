"""What a whole deep agent run looks like once it has been recorded.

The per-concern tests above check one edge each. This checks the shape of the graph they add up to, which
is the property a reader of the manifest actually depends on: everything the run produced is accounted
for by something downstream, and nothing dangles except the answer itself.
"""

from langchain_core.messages import AIMessage, HumanMessage

from scripted import call, deep_agent

LIBRARIAN = {"name": "librarian", "description": "Looks things up.", "system_prompt": "You are a librarian."}

FULL_RUN = [
    call(
        "write_todos",
        "t0",
        todos=[{"content": "research", "status": "in_progress"}, {"content": "write it up", "status": "pending"}],
    ),
    call("task", "t1", description="look up CIDs", subagent_type="librarian"),
    # the subagent's own turns
    call("write_file", "t2", file_path="/notes.md", content="A CID names content, not a location.\n"),
    AIMessage(content="Noted."),
    # back in the parent agent
    call("read_file", "t3", file_path="/notes.md"),
    call("write_file", "t4", file_path="/report.md", content="first draft\n"),
    call("edit_file", "t5", file_path="/report.md", old_string="first draft", new_string="second draft"),
    call(
        "write_todos",
        "t6",
        todos=[{"content": "research", "status": "completed"}, {"content": "write it up", "status": "completed"}],
    ),
    AIMessage(content="Here is the report."),
]


def _full_run(handler):
    return deep_agent(FULL_RUN, subagents=[LIBRARIAN]).invoke(
        {"messages": [HumanMessage("research CIDs and write a report")]},
        config={"callbacks": [handler], "recursion_limit": 80},
    )


def test_the_run_records_every_kind_of_activity(recording_handler):
    _full_run(recording_handler)

    assert {"graph", "graph_node", "agent", "tool", "chat_model"} <= recording_handler.kinds()
    assert [name for name, kind, _, _ in recording_handler.computations if kind == "agent"] == ["librarian"]


def test_nothing_the_run_produced_is_orphaned_except_its_answer(recording_handler):
    """An output no later computation consumes is a dead end in the lineage. Exactly one is expected:
    the graph's own final state, which is what the run was for."""
    _full_run(recording_handler)

    produced = {cid for _, _, _, outputs in recording_handler.computations for cid in outputs}
    consumed = {cid for _, _, inputs, _ in recording_handler.computations for cid in inputs}
    orphaned = produced - consumed

    root_output = recording_handler.outputs_of("researcher")
    assert orphaned <= set(root_output), (
        f"{len(orphaned - set(root_output))} outputs lead nowhere: "
        f"{[name for name, _, _, outs in recording_handler.computations if set(outs) & (orphaned - set(root_output))]}"
    )


def test_the_files_and_the_plan_survive_as_versioned_entities(recording_handler):
    result = _full_run(recording_handler)

    assert sorted(result["files"]) == ["/notes.md", "/report.md"]
    assert {path for path, _ in recording_handler._file_versions} == {"/notes.md", "/report.md"}
    # /report.md written once and edited once
    assert len(recording_handler._file_versions) == 3
    assert len(recording_handler._todo_versions) == 2


def test_a_subagents_file_reaches_the_parent_as_the_same_entity(recording_handler):
    """``task`` hands its state update back as a ``Command``; if that goes unclaimed the file the subagent
    wrote is re-embedded in the tool's output blob and the parent's reads point at a second copy."""
    _full_run(recording_handler)

    notes = [cid for (path, _), cid in recording_handler._file_versions.items() if path == "/notes.md"]
    assert len(notes) == 1, "one file, not one per agent that saw it"
    assert str(notes[0]) in recording_handler.inputs_of("read_file")
    assert str(notes[0]) in recording_handler.outputs_of("write_file")
