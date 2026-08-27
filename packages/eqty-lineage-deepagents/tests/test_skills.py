"""A skill the agent was given is an input to the turns that could use it.

``SkillsMiddleware`` parses each ``SKILL.md``, puts the catalogue in ``skills_metadata``, and names the
skills in the system prompt. So the skills a model turn could draw on are exactly the ones in the state
entering it -- and they are inputs the run was given, never something it produced.
"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from deepagents.middleware.skills import SkillsMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from scripted import ScriptedModel, call

SKILL = """---
name: web-research
description: Research a topic and summarise the findings.
---

Search widely, then summarise. Cite every source.
"""

SKILL_PATH = "/skills/user/web-research/SKILL.md"


def _agent(script):
    return create_deep_agent(
        model=ScriptedModel(messages=iter(script)),
        system_prompt="You are a deep research agent.",
        middleware=[SkillsMiddleware(backend=StateBackend(), sources=["/skills/user/"])],
        name="researcher",
    )


def _run(handler, script):
    return _agent(script).invoke(
        {"messages": [HumanMessage("go")], "files": {SKILL_PATH: create_file_data(SKILL)}},
        config={"callbacks": [handler], "recursion_limit": 60},
    )


def test_a_loaded_skill_is_its_own_asset(recording_handler):
    _run(recording_handler, [AIMessage(content="nothing to do")])

    names = sorted(name for name, _ in recording_handler._skill_cids)
    assert names == ["web-research"]


def test_a_skill_is_carried_into_the_model_turn_that_had_it(recording_handler):
    _run(recording_handler, [AIMessage(content="nothing to do")])

    skill = str(next(iter(recording_handler._skill_cids.values())))
    assert skill in recording_handler.inputs_of("model"), "the node whose state held the catalogue"
    produced = [name for name, _, _, outs in recording_handler.computations if skill in outs]
    assert produced == [], "a skill is given to a run, not produced by it"


def test_the_skill_file_is_registered_as_a_file_of_its_own(recording_handler):
    """The catalogue is the frontmatter; the instructions are the file, read on demand."""
    _run(recording_handler, [call("read_file", "a", file_path=SKILL_PATH), AIMessage(content="read it")])

    paths = {path for path, _ in recording_handler._file_versions}
    assert paths == {SKILL_PATH}

    version = str(next(iter(recording_handler._file_versions.values())))
    assert version in recording_handler.inputs_of("read_file")


def test_an_agent_without_skills_registers_none(recording_handler):
    from scripted import deep_agent

    deep_agent([AIMessage(content="nothing to do")]).invoke(
        {"messages": [HumanMessage("go")]},
        config={"callbacks": [recording_handler], "recursion_limit": 40},
    )

    assert recording_handler._skill_cids == {}
