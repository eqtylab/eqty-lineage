"""DeepAgents callback handler that records a deep agent run as EQTY lineage.

Attach one handler to the invocation, exactly as with the LangChain handler it subclasses::

    from eqty_lineage.deepagents import EqtyDeepAgentsHandler

    agent.invoke({"messages": [...]}, config={"callbacks": [EqtyDeepAgentsHandler()]})

Everything the LangChain handler records -- graph nodes, model calls, tool calls, retrievals, subagent
boundaries, failures -- is recorded here too. This package adds the parts of a deep agent that live in its
state rather than in its callbacks:

- every file in the virtual filesystem -> a Document asset, versioned by content and chained across edits
- every revision of the todo list     -> a Dataset asset, chained to the revision it replaced
- every loaded skill                  -> a Skill asset, carried into the model turn that could use it
- the agent and each subagent         -> an Agent asset, input to the run it identifies
- each model turn's system prompt     -> a SystemPrompt asset

``deepagents`` is not imported. Everything here is read from the callback stream and from graph state, so
the handler works against whatever version of DeepAgents produced the run.
"""

import hashlib
import json
import logging
import os.path
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.messages import BaseMessage

from eqty_sdk import CID, Agent, Dataset, Document, Skill, SystemPrompt

from eqty_lineage.langchain import UNCLAIMED, AssetSink, EqtyCallbackHandler, PathExtractor, StateExtractor, eqty_tool
from eqty_lineage.langchain._serialize import _to_jsonable

from eqty_lineage.deepagents.extractors import SkillExtractor, TodoListExtractor, VirtualFileExtractor

logger = logging.getLogger("eqty.deepagents")

#: DeepAgents filesystem tools whose arguments name a file the call reads or rewrites. The backend applies
#: its writes through LangGraph's channel API rather than through the tool's return value, so the tool
#: result never carries the file -- these are the names that say which file a call was about.
_WRITE_TOOL = "write_file"
_EDIT_TOOL = "edit_file"
_DELETE_TOOL = "delete"
_READ_TOOL = "read_file"
_FILE_TOOLS = frozenset({_WRITE_TOOL, _EDIT_TOOL, _DELETE_TOOL, _READ_TOOL})
#: runs a shell, so it can change the filesystem without naming a path
_EXECUTE_TOOL = "execute"


#: a Windows drive prefix, which the backend refuses rather than normalizes
_DRIVE_PREFIX = re.compile(r"^[a-zA-Z]:")


def _normalize_path(path: str) -> Optional[str]:
    """A virtual path in the form the filesystem actually keys it under, or None if it has no such form.

    DeepAgents puts every path through ``validate_path`` before the backend sees it, so the key in state
    is the normalized form -- a model that asks to write ``report.md`` creates ``/report.md``. Keying the
    tool's raw argument would make those two sightings two files: the write would land on one entity and
    every later read on another, and the file the run produced would be an output of nothing. Live models
    omit the leading slash routinely.

    This mirrors ``validate_path`` step for step rather than approximating it, because *near* agreement is
    the worst outcome available: two paths that the backend keeps apart but this folds together are two
    real files recorded as one asset, so a write to one is attested as a rewrite of the other. ``//x`` is
    exactly that case -- a doubled leading slash has a meaning of its own and ``normpath`` preserves
    exactly two, so normalizing it away merges a file with its neighbour. The order matters too:
    ``normpath`` runs *before* backslashes are rewritten, since on POSIX a backslash is an ordinary
    filename character until that rewrite.

    ``os.path.normpath`` for the same reason -- it is the one ``validate_path`` calls, so it is ``ntpath``
    on Windows and ``posixpath`` everywhere else, exactly as the backend's is. Hard-coding ``posixpath``
    agreed on every POSIX input and split ``.\\x`` in two on Windows, where the backend keys ``/x`` and
    this keyed ``/./x``; no test could catch it, because on POSIX the two modules are the same one.

    Where ``validate_path`` raises -- a ``..`` component, a leading ``~``, a Windows drive letter -- this
    returns None. Such a call never reaches the backend, so there is nothing to record; rewriting the path
    into something plausible instead would key the call against a file it never touched.
    """
    if ".." in PurePosixPath(path.replace("\\", "/")).parts or path.startswith("~"):
        return None
    if _DRIVE_PREFIX.match(path):
        return None

    normalized = os.path.normpath(path).replace("\\", "/")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if ".." in normalized.split("/"):
        return None
    return normalized


def _digest(value: Any) -> str:
    """A stable local key for a payload, so a repeat sighting costs no SDK registration."""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


class EqtyDeepAgentsHandler(EqtyCallbackHandler):
    """Registers a DeepAgents run as EQTY data assets and computation statements."""

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        # (path, content digest) -> Document CID. Keyed on the contents as well as the path for the same
        # reason PathExtractor is: a file rewritten mid-run is a different entity, and keying on the path
        # alone would attest content a later computation never saw.
        self._file_versions: Dict[Tuple[str, str], CID] = {}
        # path -> Document CID of its current version, so a rewrite can be chained to what it replaced
        self._file_latest: Dict[str, CID] = {}
        # path -> the current content, which is what makes an edit_file result reconstructable; see
        # _apply_edit. Held only for paths the run has actually seen.
        self._file_contents: Dict[str, str] = {}
        # plan digest -> Dataset CID, and the CID of the most recent revision
        self._todo_versions: Dict[str, CID] = {}
        self._todo_latest: Optional[CID] = None
        # (path, rendering digest) -> Document CID for a file the run only ever read; kept apart from
        # _file_versions because a read is not the file's bytes. See _record_read.
        self._read_cids: Dict[Tuple[str, str], CID] = {}
        # (name, identity digest) -> CID, so one skill or agent is registered once per run and two
        # things wearing the same name but configured differently stay two assets
        self._skill_cids: Dict[Tuple[str, str], CID] = {}
        self._agent_cids: Dict[Tuple[str, str], CID] = {}
        # prompt digest -> CID; a system prompt has no name of its own to key on
        self._system_prompt_cids: Dict[str, CID] = {}
        # tool run id -> the filesystem mutation its arguments describe, applied when the call succeeds
        self._pending_writes: Dict[UUID, Dict[str, Any]] = {}
        # root runs currently open on this handler; see _note_root
        self._open_roots: List[UUID] = []
        self._warned_about_sharing = False
        # one-shot outputs for the computation being finalized right now; see _finalize
        self._pending_outputs: List[CID] = []
        # True while a tool's arguments are being registered; see registering_tool_arguments
        self._in_tool_arguments = False

        self.add_extractor(SkillExtractor(self))
        self.add_extractor(TodoListExtractor(self))
        self.add_extractor(VirtualFileExtractor(self))

    def _note_root(self, run_id: UUID) -> None:
        """Warn once if this handler is observing two runs at the same time.

        The path and plan registries are keyed by path, or by nothing at all, because they describe one
        run's filesystem. Point a single handler at two concurrent runs and those keys collide: the second
        run's write of ``/report.md`` chains off the first run's version, and the manifest asserts that one
        run revised the other's file when they share nothing but a handler. It is a quiet failure -- the
        graph looks well-formed, it is simply wrong -- which is why it is worth a warning.

        Reusing a handler *sequentially* is not the same thing and is not flagged: across the turns of one
        conversation it is what makes a file written in the first turn and edited in the third chain
        properly, rather than appearing as two unrelated entities. The rule is one handler per
        conversation, never one shared between conversations running at once.

        Warned once per handler, since a run whose callbacks never complete would otherwise leave this
        reporting a collision on every run that follows.
        """
        if self._open_roots and not self._warned_about_sharing:
            self._warned_about_sharing = True
            logger.warning(
                "EqtyDeepAgentsHandler is observing %d runs at once. File and plan versions are keyed per "
                "run, so concurrent runs will be linked to each other's assets. Use one handler per "
                "invocation (sequential reuse across turns of one conversation is fine).",
                len(self._open_roots) + 1,
            )
        if run_id not in self._open_roots:
            self._open_roots.append(run_id)

    def _forget_root(self, run_id: UUID) -> None:
        if run_id in self._open_roots:
            self._open_roots.remove(run_id)

    @property
    def registering_tool_arguments(self) -> bool:
        """Whether the value an extractor is being offered came from a tool's arguments.

        The arguments of a call and the state it produces are serialized by the same code, so a state key
        an extractor claims is indistinguishable by key path from a tool argument of the same name --
        ``write_todos(todos=[...])`` being exactly that. Claiming it there would register the new plan as
        an *input* to the call that wrote it, which reverses the one edge the plan's lineage is for.
        """
        return self._in_tool_arguments

    ################################################## Registries ##################################################
    def register_virtual_file(
        self, path: str, content: str, metadata: Dict[str, Any]
    ) -> Tuple[Optional[CID], bool, Optional[CID]]:
        """Register one version of a virtual file, or return the version already registered.

        Returns ``(cid, created, replaced)``. ``created`` says whether this call minted a new asset;
        ``replaced`` is the version this one supersedes at that path, which the caller links as an input so
        successive edits form a chain rather than unrelated assets. The two are independent: a file
        *reverted* to bytes seen earlier mints nothing but still replaces something, and a caller that
        keyed on ``created`` alone would record the revert nowhere and leave the manifest asserting that
        the superseded content was still current.

        The path is part of the payload, so the same bytes written to two paths are two files rather than
        one entity wearing whichever name happened to be registered first.
        """
        normalized = _normalize_path(path)
        if normalized is None:
            # the backend would have refused this path, so there is no file at it to record
            logger.debug("refusing to register virtual file at an invalid path '%s'", path)
            return None, False, None
        path = normalized
        key = (path, _digest(content))
        self._file_contents[path] = content
        previous = self._file_latest.get(path)

        known = self._file_versions.get(key)
        if known is not None:
            # a sighting always shows the file as it is now, so this is the current version even when the
            # bytes are ones seen before
            self._file_latest[path] = known
            return known, False, previous if previous is not None and previous != known else None

        try:
            asset = Document.from_object(
                {"path": path, "content": content},
                name=path,
                description=f"File '{path}' in the DeepAgents virtual filesystem.",
                **metadata,
            )
        except Exception:  # noqa: BLE001 - never let the observer take down the run it observes
            logger.debug("could not register virtual file '%s'", path)
            return None, False, None

        self._file_versions[key] = asset.cid
        self._file_latest[path] = asset.cid
        return asset.cid, True, previous

    def register_todos(
        self, todos: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> Tuple[Optional[CID], bool, Optional[CID]]:
        """Register one revision of the todo list, or return the revision already registered."""
        key = _digest(todos)
        known = self._todo_versions.get(key)
        if known is not None:
            # same rule as `register_virtual_file`: a plan reverted to a revision seen before mints
            # nothing but still replaces what was current, and returning None here loses that edge
            previous = self._todo_latest
            self._todo_latest = known
            return known, False, previous if previous is not None and previous != known else None

        try:
            asset = Dataset.from_object(
                _to_jsonable(todos),
                name="todo list",
                description="The deep agent's plan, as of this revision.",
                **metadata,
            )
        except Exception:  # noqa: BLE001 - never let the observer take down the run it observes
            logger.debug("could not register a todo list revision")
            return None, False, None

        self._todo_versions[key] = asset.cid
        replaced = self._todo_latest
        self._todo_latest = asset.cid
        return asset.cid, True, replaced

    def register_skill(self, entry: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[CID]:
        """Register a loaded skill from the metadata ``SkillsMiddleware`` parsed out of its ``SKILL.md``.

        Content-addressed to that metadata, so editing a skill's frontmatter produces a different asset and
        the manifest records which version of the skill the run was given.
        """
        # redacted *after* serialization, like the system prompt: `_redact` walks only dicts and lists,
        # so an object it cannot see into passes through untouched and `_to_jsonable` then expands it
        # into a dict whose credential key would never be re-examined
        payload = self._redact(_to_jsonable(entry))
        name = str(entry.get("name") or entry.get("path") or "skill")
        key = (name, _digest(payload))
        known = self._skill_cids.get(key)
        if known is not None:
            return known

        try:
            asset = Skill.from_object(
                payload,
                name=name,
                description=str(entry.get("description") or f"Skill '{name}' loaded by the deep agent."),
                **metadata,
            )
        except Exception:  # noqa: BLE001 - never let the observer take down the run it observes
            logger.debug("could not register skill '%s'", name)
            return None

        self._skill_cids[key] = asset.cid
        return asset.cid

    def _register_agent(self, name: str, metadata: Optional[Dict[str, Any]], role: str) -> Optional[CID]:
        """Register the agent a run belongs to.

        A deep agent's root run reports its name in ``lc_agent_name`` and the library versions behind it in
        ``lc_versions``; a subagent's root run reports its own name the same way. Both are identity rather
        than position, which is why they are read directly instead of through ``_caller_metadata`` -- that
        strips every ``lc_*`` key, since most of them do encode a position.
        """
        meta = metadata or {}
        payload: Dict[str, Any] = {"agent": name, "role": role, "framework": self._framework or "deepagents"}
        versions = meta.get("lc_versions")
        if isinstance(versions, dict) and versions:
            payload["versions"] = _to_jsonable(versions)
        config = self._caller_metadata(meta)
        if config:
            payload["config"] = _to_jsonable(config)

        key = (name, _digest(payload))
        known = self._agent_cids.get(key)
        if known is not None:
            return known

        try:
            asset = Agent.from_object(
                payload,
                name=name,
                description=f"The {role} '{name}'.",
                **self._verbose_metadata({"callback": "on_chain_start", "metadata": metadata}),
            )
        except Exception:  # noqa: BLE001 - never let the observer take down the run it observes
            logger.debug("could not register agent '%s'", name)
            return None

        self._agent_cids[key] = asset.cid
        return asset.cid

    def _register_system_prompt(self, messages: List[List[BaseMessage]], model_name: str) -> Optional[CID]:
        """Register the system prompt a model turn was given, if it had one.

        The deep agent's prompt is not the string the caller passed: the middleware stack appends its own
        sections to it -- the skills catalogue, the todo instructions -- so what the model was actually
        told is only visible here. Content-addressed, so the parent agent's prompt and each subagent's are
        distinct assets and an unchanged prompt is one asset across the whole run.
        """
        system = next(
            (message for batch in messages for message in batch if getattr(message, "type", None) == "system"),
            None,
        )
        if system is None:
            return None

        # redacted like every other payload that reaches an asset: a middleware is free to interpolate
        # configured credentials into the prompt it assembles, and these are stored as blobs on disk
        payload = self._redact(_to_jsonable(system.content))
        key = _digest(payload)
        known = self._system_prompt_cids.get(key)
        if known is not None:
            return known

        try:
            asset = SystemPrompt.from_object(
                payload,
                name=f"{model_name}: system prompt",
                description="System prompt the deep agent's middleware stack assembled for this turn.",
                **self._verbose_metadata({"callback": "on_chat_model_start", "model": model_name}),
            )
        except Exception:  # noqa: BLE001 - never let the observer take down the run it observes
            logger.debug("could not register the system prompt for '%s'", model_name)
            return None

        self._system_prompt_cids[key] = asset.cid
        return asset.cid

    ################################################## Registries ##################################################

    def _finalize(self, name: str, kind: str, input_cids: List[CID], output_cids: List[CID]) -> None:
        """Fold in any outputs the callback that is finalizing could not put in ``output_cids`` itself.

        A file written by a tool is one: the DeepAgents state backend applies writes through LangGraph's
        channel API, so the new content is in neither the tool's result nor the enclosing node's output
        state, and the base handler has nowhere to hang it. ``on_tool_end`` reconstructs it and leaves it
        here for the ``_finalize`` its own ``super()`` call is about to make -- a single hand-off, made
        under the handler's lock and cleared in a ``finally``, so it can neither race another callback nor
        leak into the next computation if registration raises.
        """
        if self._pending_outputs:
            pending, self._pending_outputs = self._pending_outputs, []
            # A version already among the inputs is not added as an output, which would make the
            # computation its own ancestor. No shipped backend reaches this: `write_file` returns a plain
            # `ToolMessage`, not a `Command` carrying the new content, and a rewrite with unchanged bytes
            # registers nothing to hand over. It guards a backend that echoes its write back.
            #
            # Note what it does *not* do: the collision is settled in favour of the input, so if this ever
            # did fire the write's own product would stay an input to the write. For a backend that echoes,
            # dropping it from the inputs and keeping it as an output is the truthful resolution -- worth
            # revisiting with a backend in hand rather than guessing at one.
            output_cids = [
                *output_cids,
                *(cid for cid in pending if cid not in output_cids and cid not in input_cids),
            ]
        return super()._finalize(name, kind, input_cids, output_cids)

    ################################################## Chain Calls #################################################
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Link the agent a tracked run belongs to into that run's inputs.

        The base handler already decides which runs are worth tracking and which of them are subagent
        boundaries, so this reads its decision back off the run rather than repeating the rule.
        """
        # held across the `super()` call for the same reason as `on_tool_start`: the run this reads back
        # is the one that call created, and a concurrent sibling must not be able to act between them
        with self._lock:
            super().on_chain_start(
                serialized,
                inputs,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )

            if parent_run_id is None:
                self._note_root(run_id)

            run = self._runs.get(run_id)
            if run is None or run["kind"] not in ("graph", "agent"):
                return

            role = "deep agent" if run["kind"] == "graph" else "subagent"
            agent_cid = self._register_agent(run["name"], metadata, role)
            if agent_cid is None:
                return
            if agent_cid not in run["inputs"]:
                run["inputs"].append(agent_cid)

            if run["kind"] == "agent":
                # the subagent spec is also an input to the `task` call that asked for it, which would
                # otherwise show a subagent's whole run appearing from a tool call that named nothing.
                # Only when the enclosing run is a tool: `_enclosing_node` returns the nearest tracked run
                # of any kind, so a subagent reached other than through a tool -- a subgraph wired in
                # directly -- would otherwise attach its identity to a node that never invoked it.
                spawning_tool = self._enclosing_node(parent_run_id)
                if spawning_tool is not None and spawning_tool.get("kind") == "tool":
                    if agent_cid not in spawning_tool["inputs"]:
                        spawning_tool["inputs"].append(agent_cid)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            self._forget_root(run_id)
            super().on_chain_end(outputs, run_id=run_id, **kwargs)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            self._forget_root(run_id)
            super().on_chain_error(error, run_id=run_id, **kwargs)

    ################################################## Chain Calls #################################################

    ################################################## LLM Calls ###################################################
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Register the system prompt this turn was given as an input of its own."""
        super().on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            prompt_cid = self._register_system_prompt(messages, run["name"])
            if prompt_cid is not None and prompt_cid not in run["inputs"]:
                run["inputs"].append(prompt_cid)

    ################################################## LLM Calls ###################################################

    ################################################## Tool Calls ##################################################
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Link the file a filesystem tool was pointed at, and remember what the call is about to change.

        The file a call reads or rewrites is an input to it: the version being replaced is what the edit
        was made against, and the version being read is what the model went on to reason from. Neither is
        visible in the tool's arguments as an asset -- the arguments name a path -- so the link is made
        from the version registry instead.
        """
        # one critical section, not two: LangGraph runs the tool calls of a single AI message concurrently
        # on a thread pool, even under plain `.invoke()`. Releasing the lock between registering the
        # arguments and reading the version registry lets a sibling call's write land in between, and the
        # read is then attested against content it never saw. The lock is reentrant, so the `super()` call
        # taking it again is free.
        with self._lock:
            self._in_tool_arguments = True
            try:
                super().on_tool_start(
                    serialized,
                    input_str,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    tags=tags,
                    metadata=metadata,
                    inputs=inputs,
                    **kwargs,
                )
            finally:
                self._in_tool_arguments = False

            tool_name = (serialized or {}).get("name", "tool")
            if tool_name == _EXECUTE_TOOL:
                self._pending_writes[run_id] = {"executed": True}
                return
            if tool_name not in _FILE_TOOLS or not isinstance(inputs, dict):
                return
            raw_path = inputs.get("file_path")
            if not isinstance(raw_path, str) or not raw_path:
                return
            path = _normalize_path(raw_path)
            if path is None:
                # a path the backend refuses never reaches it, so the call touches no file: linking one
                # here would put an edge to a file this call never read or replaced
                return

            run = self._runs.get(run_id)
            if run is None:
                return

            current = self._file_latest.get(path)
            if current is not None and current not in run["inputs"]:
                run["inputs"].append(current)

            if tool_name == _READ_TOOL:
                # `read_file` takes offset/limit, so two calls can return different windows of one file.
                # Without the window in the payload they are two assets wearing the same name.
                window = {k: inputs[k] for k in ("offset", "limit") if isinstance(inputs.get(k), int)}
                self._pending_writes[run_id] = {"path": path, "read": True, "window": window}
            elif tool_name == _DELETE_TOOL:
                self._pending_writes[run_id] = {"path": path, "deleted": True}
            elif tool_name == _WRITE_TOOL and isinstance(inputs.get("content"), str):
                self._pending_writes[run_id] = {"path": path, "content": inputs["content"]}
            elif tool_name == _EDIT_TOOL and isinstance(inputs.get("old_string"), str):
                self._pending_writes[run_id] = {
                    "path": path,
                    "old_string": inputs["old_string"],
                    "new_string": inputs.get("new_string") or "",
                    "replace_all": bool(inputs.get("replace_all")),
                }

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Record the file a successful filesystem call wrote, as an output of that call.

        Without this a written file is an input to every computation that later reads it and an output of
        none, which in a provenance graph says the run found it already there. It says that because the
        DeepAgents state backend writes through LangGraph's channel API: the content reaches ``files`` in
        state without ever passing through the tool's result, so the only place it can be attributed to
        the call that produced it is here, from the arguments the call was made with.
        """
        with self._lock:
            pending = self._pending_writes.pop(run_id, None)
            try:
                if pending is not None:
                    self._record_write(run_id, pending, output)
                super().on_tool_end(output, run_id=run_id, **kwargs)
            finally:
                # cleared unconditionally: if registration raised, the hand-off must not survive into
                # whichever computation is finalized next
                self._pending_outputs = []

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._pending_writes.pop(run_id, None)
        super().on_tool_error(error, run_id=run_id, **kwargs)

    def _record_write(self, run_id: UUID, pending: Dict[str, Any], output: Any) -> None:
        """Register what a successful filesystem call did, and hand it to the finalize that follows."""
        if _is_tool_error(output):
            return

        if pending.get("executed"):
            # A shell can create, rewrite or remove files, and the command says nothing about which -- so
            # what it did cannot be recorded. What can be avoided is *asserting* the filesystem it left
            # behind: the reconstruction caches are dropped, so a later `edit_file` reconstructs against
            # nothing and registers nothing, and a later `read_file` registers the rendering it actually
            # got instead of linking a version that may no longer exist. A gap where a shell ran, rather
            # than a version the file never held.
            #
            # `execute` only reaches here on success, and only a sandbox or local-shell backend implements
            # it -- every other backend errors the call -- so this does not fire for the state and
            # filesystem backends, where the shell and the agent's filesystem are not the same thing
            # anyway. Under the state backend the extractor re-registers everything from the next state it
            # sees, so nothing is lost there either.
            self._file_latest.clear()
            self._file_contents.clear()
            return

        path = pending["path"]
        if pending.get("read"):
            self._record_read(run_id, path, output, pending.get("window") or {})
            return

        if pending.get("deleted"):
            # the file is gone; keeping its last version as "current" would chain a later write to
            # content that no longer existed, and link a later read to a version it could not have read.
            # `delete` takes a directory too: the backend drops the exact key and everything under
            # `base + "/"`, so forgetting only `path` leaves every nested file current. The prefix needs
            # that trailing slash -- `/dirx.md` starts with `/dir` but is not under it.
            base = path.rstrip("/")
            prefix = f"{base}/"
            for known in [k for k in {*self._file_latest, *self._file_contents} if k == base or k.startswith(prefix)]:
                self._file_latest.pop(known, None)
                self._file_contents.pop(known, None)
            return

        if "content" in pending:
            content = pending["content"]
        else:
            content = self._apply_edit(path, pending)
            if content is None:
                return

        cid, created, replaced = self.register_virtual_file(
            path,
            content,
            self._verbose_metadata({"callback": "on_tool_end", "run_id": run_id, "file_path": path}),
        )
        if cid is None:
            return
        # `created` alone is not the test: a file reverted to bytes seen earlier mints no new asset but
        # is still a write, and skipping it would leave the manifest asserting that the version it
        # replaced was still the current one
        if not created and replaced is None:
            return

        run = self._runs.get(run_id)
        if run is not None and replaced is not None and replaced not in run["inputs"]:
            run["inputs"].append(replaced)
        self._pending_outputs.append(cid)

    def _record_read(self, run_id: UUID, path: str, output: Any, window: Dict[str, int]) -> None:
        """Give a file the run only ever *read* a place in the graph.

        A file the agent writes is registered from the write's arguments, and one carried in graph state
        is registered by the extractor. Neither reaches a file that already existed and is never written --
        which is every source file an agent reads under a filesystem, store or sandbox backend, since those
        keep the filesystem out of state entirely. Left alone, the model's answer derives from a `read_file`
        computation that consumed nothing, and the file it was actually built from is absent.

        What is recorded is what the tool returned, and it is labelled as such rather than as the file: the
        result is a *rendering* -- line-numbered, chunked at long lines, truncated when large -- and it is
        lossy in a way that cannot be undone. A file ending in a newline renders identically to one that
        does not, so reconstructing the bytes is impossible, not merely fragile. Recording the rendering
        under the file's own identity would therefore assert a content hash the file never had.

        Kept out of `_file_versions` and `_file_contents` for the same reason: a rendering must never
        become the version a later `edit_file` is reconstructed against, nor the version a later read is
        linked to. It is an *input* -- the run consumed it and did not produce it -- and only ever when the
        path has no real version already, so a file the run wrote is never shadowed by how it was read.

        `read_file` takes `offset` and `limit`, so one file can be read in several windows. Each is its own
        rendering, and the window is part of the payload and of the name: without it a manifest holds two
        assets both called `/big.md` and both described as the file, with nothing saying either is a slice
        of it -- and two windows that happened to render alike would collapse into one.
        """
        if self._file_latest.get(path) is not None:
            return  # on_tool_start already linked the version this read saw

        content = getattr(output, "content", output)
        if not isinstance(content, str) or not content:
            return

        payload: Dict[str, Any] = {"path": path, "read": content}
        if window:
            payload["window"] = dict(sorted(window.items()))
        slice_of = "".join(f", {name} {value}" for name, value in sorted(window.items()))
        key = (path, _digest(payload))
        cid = self._read_cids.get(key)
        if cid is None:
            try:
                asset = Document.from_object(
                    payload,
                    name=f"{path} ({slice_of.lstrip(', ')})" if window else path,
                    description=(
                        f"File '{path}', as the deep agent read it{slice_of}. The tool's rendering of the "
                        f"file rather than its bytes: the run never wrote this path, so its content was "
                        f"never observable exactly."
                    ),
                    **self._verbose_metadata({"callback": "on_tool_end", "run_id": run_id, "file_path": path}),
                )
            except Exception:  # noqa: BLE001 - never let the observer take down the run it observes
                logger.debug("could not register the read of '%s'", path)
                return
            self._read_cids[key] = asset.cid
            cid = asset.cid

        run = self._runs.get(run_id)
        if run is not None and cid not in run["inputs"]:
            run["inputs"].append(cid)

    def _apply_edit(self, path: str, pending: Dict[str, Any]) -> Optional[str]:
        """The content ``edit_file`` produced, or ``None`` when that cannot be known exactly.

        ``edit_file`` reports only that it succeeded, so the resulting content has to be derived from the
        version the edit was made against. That is a plain string replacement -- and the same occurrence
        rules the backend applies are re-checked here, so a case where this would be guessing produces
        nothing rather than a file version the run never had. The file then registers at its next sighting
        in state, as an input, which understates its provenance but does not misstate it.
        """
        current = self._file_contents.get(path)
        if current is None:
            return None
        old = pending["old_string"]
        occurrences = current.count(old)
        if occurrences == 0 or (occurrences > 1 and not pending["replace_all"]):
            return None
        return current.replace(old, pending["new_string"])

    ################################################## Tool Calls ##################################################


def _is_tool_error(output: Any) -> bool:
    """Whether a filesystem tool reported a failure in its result rather than by raising.

    DeepAgents' filesystem tools return their errors as ordinary text -- a missing string, an ambiguous
    match, a backend that was unreachable -- so a call that "succeeded" as far as the callbacks are
    concerned may have changed nothing.

    ``status`` is the reliable signal and is checked first: the wording is the backend's own, and only
    some of them say "Error" (the store, LangSmith and sandbox backends variously report "Failed to write
    file ..." or the remote's message verbatim). Reading the text alone would take those for successes and
    attest a file version that was never written. The prefix is still honoured, for a result that carries
    no status at all.
    """
    if getattr(output, "status", None) == "error":
        return True
    content = getattr(output, "content", output)
    return isinstance(content, str) and content.lstrip().startswith("Error")


__all__ = [
    "UNCLAIMED",
    "AssetSink",
    "EqtyCallbackHandler",
    "EqtyDeepAgentsHandler",
    "PathExtractor",
    "SkillExtractor",
    "StateExtractor",
    "TodoListExtractor",
    "VirtualFileExtractor",
    "eqty_tool",
]
