"""A deep agent pausing for approval is not a deep agent that failed.

``create_deep_agent(interrupt_on=...)`` installs ``HumanInTheLoopMiddleware``, which calls
``interrupt()`` before a matched tool runs. That raises ``GraphInterrupt`` to suspend the graph, and it
reaches ``on_chain_error`` looking exactly like a crash.

The guard that tells the two apart lives in :mod:`eqty_lineage.langchain` (released in
``eqty-lineage-langchain@0.1.1``), and this handler inherits it: :meth:`on_chain_error` here releases the
run's root and then delegates. These tests exist because *inheriting* a fix is a claim about this
package, not about the one that shipped it -- a future override that forgot to call ``super()`` would
reopen the bug here while the LangChain tests stayed green.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from scripted import ScriptedModel, call

APPROVE = {"decisions": [{"type": "approve"}]}


def _approving_agent(script):
    """A deep agent that must have ``write_file`` approved before it runs.

    A checkpointer is required: an interrupt suspends the graph, and without somewhere to persist that
    suspension there is nothing to resume into.
    """
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=ScriptedModel(messages=iter(script)),
        system_prompt="Write the file you are asked for.",
        interrupt_on={"write_file": True},
        checkpointer=MemorySaver(),
        name="approver",
    )


SCRIPT = [
    call("write_file", "w1", file_path="/report.md", content="# Report\n"),
    AIMessage(content="Written."),
]


def test_pausing_for_approval_records_no_failure(recording_handler):
    """The turn that pauses records nothing, rather than a failure that did not happen."""
    agent = _approving_agent(SCRIPT)
    config = {"callbacks": [recording_handler], "configurable": {"thread_id": "t1"}, "recursion_limit": 60}

    result = agent.invoke({"messages": [HumanMessage("write the report")]}, config=config)

    assert "__interrupt__" in result, "the agent did not actually pause; the test proves nothing"
    kinds = [kind for _name, kind, _ins, _outs in recording_handler.computations]
    assert not [k for k in kinds if k.endswith("_error")], f"a pause was recorded as a failure: {kinds}"


def test_resuming_records_the_work_once(recording_handler):
    """Approving the call records the tool run itself, exactly once and not as an error."""
    agent = _approving_agent(SCRIPT)
    config = {"callbacks": [recording_handler], "configurable": {"thread_id": "t2"}, "recursion_limit": 60}

    agent.invoke({"messages": [HumanMessage("write the report")]}, config=config)
    agent.invoke(Command(resume=APPROVE), config=config)

    names = [name for name, _kind, _ins, _outs in recording_handler.computations]
    kinds = [kind for _name, kind, _ins, _outs in recording_handler.computations]
    assert not [k for k in kinds if k.endswith("_error")], f"resuming recorded a failure: {kinds}"
    assert names.count("write_file") == 1, f"write_file recorded {names.count('write_file')} times: {names}"
