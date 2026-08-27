"""Each revision of the plan is its own asset, chained to the one it replaced.

A plan written once and never revised is one asset; a plan revised three times is three, in a chain. The
alternative -- one "todos" blob re-serialized into every node's state -- records that the agent had a plan
but never what the plan was at any point, which is the only part worth attesting.

``TodoListMiddleware`` is not in the default deep agent stack: it comes from ``langchain`` and has to be
passed explicitly, which is why ``scripted.deep_agent`` adds it.
"""

from langchain_core.messages import AIMessage, HumanMessage

from scripted import call, deep_agent

PLAN_A = [{"content": "research the topic", "status": "in_progress"}, {"content": "draft", "status": "pending"}]
PLAN_B = [{"content": "research the topic", "status": "completed"}, {"content": "draft", "status": "in_progress"}]
PLAN_C = [{"content": "research the topic", "status": "completed"}, {"content": "draft", "status": "completed"}]


def _run(handler, script, todos=True):
    return deep_agent(script, todos=todos).invoke(
        {"messages": [HumanMessage("plan and do the work")]},
        config={"callbacks": [handler], "recursion_limit": 60},
    )


def test_each_revision_is_its_own_asset(recording_handler):
    _run(
        recording_handler,
        [
            call("write_todos", "a", todos=PLAN_A),
            call("write_todos", "b", todos=PLAN_B),
            call("write_todos", "c", todos=PLAN_C),
            AIMessage(content="done"),
        ],
    )

    assert len(recording_handler._todo_versions) == 3


def test_a_revision_is_an_output_of_the_call_that_wrote_it(recording_handler):
    _run(recording_handler, [call("write_todos", "a", todos=PLAN_A), AIMessage(content="done")])

    written = str(next(iter(recording_handler._todo_versions.values())))
    assert written in recording_handler.outputs_of("write_todos")


def test_revisions_are_chained(recording_handler):
    _run(
        recording_handler,
        [
            call("write_todos", "a", todos=PLAN_A),
            call("write_todos", "b", todos=PLAN_B),
            AIMessage(content="done"),
        ],
    )

    first, second = (str(cid) for cid in recording_handler._todo_versions.values())
    revision = [c for c in recording_handler.computations if c[0] == "write_todos"][1]
    assert first in revision[2], "the revision it replaced is what the new plan was written against"
    assert second in revision[3]


def test_an_unrevised_plan_stays_one_asset(recording_handler):
    """The plan is in the state of every model turn after it is written."""
    _run(
        recording_handler,
        [
            call("write_todos", "a", todos=PLAN_A),
            call("read_file", "b", file_path="/nothing.md"),
            AIMessage(content="done"),
        ],
    )

    assert len(recording_handler._todo_versions) == 1
    only = str(next(iter(recording_handler._todo_versions.values())))
    produced = [name for name, _, _, outs in recording_handler.computations if only in outs]
    assert produced == ["write_todos"]


def test_an_agent_without_the_middleware_records_no_plan(recording_handler):
    """The default deep agent has neither the tool nor the state key, and must not grow a phantom one."""
    _run(recording_handler, [AIMessage(content="nothing to plan")], todos=False)

    assert recording_handler._todo_versions == {}
