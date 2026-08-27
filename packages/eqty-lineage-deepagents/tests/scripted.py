"""Building a deep agent that runs off a script, so a test needs no API key and never varies."""

from typing import Any, Dict, List, Optional

from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed list of messages and accepts a tool belt without binding it."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def call(name: str, call_id: str, **args: Any) -> AIMessage:
    """An assistant turn that asks for exactly one tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "id": call_id, "args": args}])


def deep_agent(
    script: List[AIMessage],
    *,
    subagents: Optional[List[Dict[str, Any]]] = None,
    todos: bool = True,
    system_prompt: str = "You are a deep research agent.",
    name: str = "researcher",
) -> Any:
    """A deep agent whose every turn is scripted.

    ``TodoListMiddleware`` is passed explicitly because it is not part of the default deep agent stack --
    it comes from ``langchain``, and a stock ``create_deep_agent`` has neither ``write_todos`` nor the
    ``todos`` state key.
    """
    return create_deep_agent(
        model=ScriptedModel(messages=iter(script)),
        system_prompt=system_prompt,
        subagents=subagents or [],
        middleware=[TodoListMiddleware()] if todos else [],
        name=name,
    )
