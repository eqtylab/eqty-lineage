"""The agent, its subagents, and the prompt each model turn was actually given.

A deep agent's run is otherwise a graph computation attributed to nothing: the thing that ran leaves no
asset behind, and neither does the subagent a ``task`` call delegated to. The system prompt matters for
the same reason -- what the model was told is not the string the caller passed, because the middleware
stack appends the skills catalogue and the todo instructions to it.
"""

from langchain_core.messages import AIMessage, HumanMessage

from scripted import call, deep_agent

LIBRARIAN = {"name": "librarian", "description": "Looks things up.", "system_prompt": "You are a librarian."}


def _run(handler, script, **kwargs):
    return deep_agent(script, **kwargs).invoke(
        {"messages": [HumanMessage("go")]},
        config={"callbacks": [handler], "recursion_limit": 60},
    )


def test_the_deep_agent_is_an_asset_input_to_its_own_run(recording_handler):
    _run(recording_handler, [AIMessage(content="nothing to do")])

    assert len(recording_handler._agent_cids) == 1
    agent_cid = str(next(iter(recording_handler._agent_cids.values())))
    assert agent_cid in recording_handler.inputs_of("researcher")


def test_a_subagent_is_its_own_asset_and_feeds_the_task_that_spawned_it(recording_handler):
    _run(
        recording_handler,
        [
            call("task", "a", description="look it up", subagent_type="librarian"),
            AIMessage(content="found it"),
            AIMessage(content="done"),
        ],
        subagents=[LIBRARIAN],
    )

    assert len(recording_handler._agent_cids) == 2, "the deep agent and its subagent are different agents"
    subagent = str(_agent_asset(recording_handler, "librarian"))

    assert subagent in recording_handler.inputs_of("librarian"), "the subagent identifies its own run"
    assert subagent in recording_handler.inputs_of("task"), (
        "a task call that names no subagent asset produces a whole delegated run out of nothing"
    )
    assert "agent" in recording_handler.kinds()


def test_the_system_prompt_is_registered_once_per_distinct_prompt(recording_handler):
    """The prompt is rebuilt for every turn, but an unchanged prompt is one asset across the run."""
    _run(
        recording_handler,
        [
            call("read_file", "a", file_path="/nothing.md"),
            AIMessage(content="done"),
        ],
    )

    assert len(recording_handler._system_prompt_cids) == 1
    prompt = str(next(iter(recording_handler._system_prompt_cids.values())))
    assert prompt in recording_handler.inputs_of("ScriptedModel")


def test_a_subagent_prompt_is_a_different_asset(recording_handler):
    _run(
        recording_handler,
        [
            call("task", "a", description="look it up", subagent_type="librarian"),
            AIMessage(content="found it"),
            AIMessage(content="done"),
        ],
        subagents=[LIBRARIAN],
    )

    assert len(recording_handler._system_prompt_cids) == 2, (
        "the subagent was told something different, so it was given a different prompt"
    )


def test_the_framework_is_read_from_the_run(recording_handler):
    _run(recording_handler, [AIMessage(content="nothing to do")])
    assert recording_handler._framework == "deepagents"


def _agent_asset(handler, name: str):
    """The Agent asset registered for ``name``."""
    for (agent, _digest), cid in handler._agent_cids.items():
        if agent == name:
            return cid
    raise AssertionError(f"no agent asset for '{name}'; have {sorted(a for a, _ in handler._agent_cids)}")
