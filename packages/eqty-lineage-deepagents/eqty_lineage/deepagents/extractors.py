"""Extractors that lift DeepAgents' own state keys out of the bulk state blob.

A deep agent carries its virtual filesystem, its plan and its skill catalogue in graph state, and every
node's state is otherwise serialized whole into that node's Dataset asset. Without these, a run's
filesystem is re-embedded once per node and no file, plan revision or skill is ever an entity in its own
right.

Each extractor claims one state key, hands the value to the handler's registry -- which owns the
carried-versus-created decision and the version chain -- and returns a compact stand-in for the payload.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from eqty_lineage.langchain import UNCLAIMED, AssetSink, StateExtractor

if TYPE_CHECKING:
    from eqty_lineage.deepagents import EqtyDeepAgentsHandler

logger = logging.getLogger("eqty.deepagents")


def _file_content(data: Any) -> Optional[str]:
    """The text of one entry in the ``files`` state, or None if it has none to read.

    DeepAgents still accepts a legacy ``list[str]`` content and joins it on newlines in
    ``file_data_to_string``; a checkpoint written by an older version replays that shape. Reading only
    ``str`` would drop the whole filesystem of a resumed run out of the manifest.
    """
    content = data.get("content") if isinstance(data, dict) else data
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(line, str) for line in content):
        return "\n".join(content)
    return None


def _state_paths(key: str) -> frozenset:
    """Where a state key legitimately appears, as opposed to where it merely looks like it does.

    A state key is claimed at the top of a state, and inside the ``update`` of a ``Command`` -- which is
    how a tool applies one, and how ``task`` hands a subagent's work back. Claiming the key anywhere it
    turns up would also claim it inside an assistant message's ``tool_calls``, where the same name is the
    argument the model *asked* for rather than the state that resulted: the request would be attributed to
    the model turn that made it, and the tool call that applied it would produce nothing.
    """
    return frozenset({(key,), ("update", key)})


class _HandlerExtractor(StateExtractor):
    """Base for the extractors below, all of which delegate to the handler's asset registries."""

    #: the key paths in graph state this extractor answers to; see _state_paths
    KEY_PATHS: frozenset = frozenset()

    def __init__(self, handler: "EqtyDeepAgentsHandler") -> None:
        self._handler = handler

    def _claims(self, key_path: Tuple[str, ...]) -> bool:
        """Whether this key path is one of ours, and is graph state rather than a tool's arguments."""
        return key_path in self.KEY_PATHS and not self._handler.registering_tool_arguments


class VirtualFileExtractor(_HandlerExtractor):
    """Registers each file in the DeepAgents virtual filesystem as a ``Document`` of its own.

    Claims the ``files`` state key -- the key the ``StateBackend`` writes to -- both at the top of a state
    and inside the ``Command`` update that ``task`` returns from a subagent. Both are the same filesystem
    seen from different places, so both are claimed: leaving the ``Command`` unclaimed would embed every
    file the subagent touched in the ``task`` tool's output blob.

    Versions are keyed on ``(path, content)``, exactly as :class:`~eqty_lineage.langchain.PathExtractor`
    keys real paths on ``(path, content CID)``. Same bytes at the same path are one entity carried through;
    new bytes are a new version, with the one it replaced linked as an input.
    """

    #: see _state_paths
    KEY_PATHS = _state_paths("files")

    def extract(self, key_path: Tuple[str, ...], value: Any, sink: AssetSink) -> Any:
        if not self._claims(key_path) or not isinstance(value, dict):
            return UNCLAIMED

        replacement: Dict[str, str] = {}
        for path, data in sorted(value.items(), key=lambda item: str(item[0])):
            if data is None:
                # The backend queues a None into the files channel to mark a deletion, but the channel's
                # reducer pops the key rather than storing the marker -- so the state handed to an
                # extractor never carries one, and this has no effect on the registries by design. It
                # covers a raw delta arriving in a `Command` update, and a reducer that stops popping.
                replacement[str(path)] = "<deleted>"
                continue
            content = _file_content(data)
            if content is None:
                # a binary file whose content the backend has not decoded, or a shape this version of
                # DeepAgents does not use; recorded as present rather than guessed at
                logger.debug("no readable content for virtual file '%s'", path)
                replacement[str(path)] = "<unreadable>"
                continue
            cid, created, replaced = self._handler.register_virtual_file(str(path), content, sink.metadata)
            if cid is None:
                replacement[str(path)] = "<unregistered>"
                continue
            if created or replaced is not None:
                # `created` alone is not the test: state showing the file back at bytes seen earlier
                # mints nothing, but it is still what this node left behind, and keying on `created`
                # would record the revert nowhere -- the mistake `register_virtual_file` warns about
                if replaced is not None:
                    sink.carry(replaced)
                sink.create(cid)
            else:
                sink.carry(cid)
            replacement[str(path)] = str(cid)

        # the paths and the CID of each one's current content: enough to identify what the state held
        # without re-embedding the filesystem in every node's state asset
        return replacement


class TodoListExtractor(_HandlerExtractor):
    """Registers each revision of the agent's plan as a ``Dataset`` of its own.

    ``TodoListMiddleware`` is not part of the default deep agent stack -- it comes from ``langchain`` and
    has to be passed explicitly -- so this claims the key only when it is actually there.

    ``write_todos`` returns a ``Command`` whose update carries the new list, which is why a revision is
    created as an output of the tool call that wrote it rather than merely appearing in the next node's
    state. The revision it replaced is linked as an input, so a plan revised four times is four assets in
    a chain rather than four unrelated ones.
    """

    #: see _state_paths
    KEY_PATHS = _state_paths("todos")

    def extract(self, key_path: Tuple[str, ...], value: Any, sink: AssetSink) -> Any:
        if not self._claims(key_path) or not isinstance(value, list):
            return UNCLAIMED
        if not all(isinstance(item, dict) and "content" in item for item in value):
            return UNCLAIMED

        cid, created, replaced = self._handler.register_todos(value, sink.metadata)
        if cid is None:
            return UNCLAIMED
        if created or replaced is not None:
            # see VirtualFileExtractor: a revert mints nothing but still supersedes a revision
            if replaced is not None:
                sink.carry(replaced)
            sink.create(cid)
        else:
            sink.carry(cid)

        # the plan is its own asset now; the state keeps a summary so a reader can still see where the
        # run had got to without the full list being duplicated into every node
        return {"todos": str(cid), "open": sum(1 for item in value if item.get("status") != "completed")}


class SkillExtractor(_HandlerExtractor):
    """Registers each loaded skill as a ``Skill`` asset, carried into the computation that had it.

    ``SkillsMiddleware`` puts the catalogue it loaded in ``skills_metadata`` and names each skill in the
    system prompt, so the skills a model turn could draw on are exactly the ones in the state entering it.
    They are *carried*, never created: a skill is an input the run was given, and the ``SKILL.md`` file it
    was parsed from is registered separately by :class:`VirtualFileExtractor` when it is read.
    """

    #: see _state_paths
    KEY_PATHS = _state_paths("skills_metadata")

    def extract(self, key_path: Tuple[str, ...], value: Any, sink: AssetSink) -> Any:
        if not self._claims(key_path) or not isinstance(value, list):
            return UNCLAIMED

        # validated in full before anything is registered. Declining re-embeds the catalogue in the state
        # blob whichever point it happens at, so that is not what decides this; declining *halfway* also
        # leaves the entries seen so far linked as carried inputs, and a turn recorded as having been
        # given three of five skills reads as fact rather than as the gap it is
        if not all(isinstance(entry, dict) for entry in value):
            return UNCLAIMED

        names: List[str] = []
        for entry in value:
            cid = self._handler.register_skill(entry, sink.metadata)
            if cid is not None:
                sink.carry(cid)
            names.append(str(entry.get("name") or entry.get("path") or "skill"))

        return sorted(names)
