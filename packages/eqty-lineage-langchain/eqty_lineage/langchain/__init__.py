"""LangChain/LangGraph callback handler that registers compute and data with the eqty_sdk.

Instead of decorating every graph node with ``@eqty_sdk.compute``, attach a single ``EqtyCallbackHandler``
to the graph invocation::

    app.invoke(state, config={"callbacks": [EqtyCallbackHandler()]})

The handler listens to the runs LangGraph emits and turns them into EQTY lineage:

- every graph node run   -> input/output Dataset assets + a computation statement
- every chat model call  -> Prompt + Model assets in, Reasoning asset out + computation
- every tool call        -> Tool + input Dataset in, output Dataset out + computation

``pathlib.Path`` values in graph state get special treatment: if the path exists on disk, the file or directory it
points to is registered as its own Dataset asset via ``Dataset.from_path`` (CIDing the full directory contents), and
that asset is linked into the computation that carried the path.

That treatment is one :class:`StateExtractor` -- :class:`PathExtractor` -- and more can be registered with
``add_extractor``. An extractor claims part of a state, registers whatever assets represent it, and replaces it in the
bulk state blob, which is how a framework package teaches this handler about its own state without this package
having to know the framework exists.
"""

import functools
import inspect
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from eqty_sdk import CID, Dataset, Document, Model, Prompt, Reasoning, Tool, get_cid_for_path
from eqty_sdk.metadata import Metadata
from eqty_sdk.statements import add_computation_statement

logger = logging.getLogger("eqty.langgraph")


#: Returned by a claim hook that does not want the value; a plain None cannot serve, because None is
#: itself a legitimate replacement.
UNCLAIMED = object()


def _to_jsonable(
    obj: Any,
    on_value: Optional[Callable[[Tuple[str, ...], Any], Any]] = None,
    _key_path: Tuple[str, ...] = (),
) -> Any:
    """Convert LangChain/LangGraph values into plain JSON-serializable data.

    ``on_value`` is offered every value encountered, with the sequence of dict keys that led to it. It
    returns ``UNCLAIMED`` to decline, or a replacement to substitute into the payload -- which is how a
    :class:`StateExtractor` lifts something out of the bulk state blob and into an asset of its own.
    """
    if on_value is not None:
        claimed = on_value(_key_path, obj)
        if claimed is not UNCLAIMED:
            return claimed
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseMessage):
        data: Dict[str, Any] = {"role": obj.type, "content": _to_jsonable(obj.content, on_value, _key_path)}
        tool_calls = getattr(obj, "tool_calls", None)
        if tool_calls:
            data["tool_calls"] = _to_jsonable(tool_calls, on_value, _key_path)
        usage = getattr(obj, "usage_metadata", None)
        if usage:
            data["usage"] = _to_jsonable(usage, on_value, _key_path)
        return data
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, on_value, (*_key_path, str(k))) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # the index is not part of the key path: an extractor claims a state key, not a position in a list
        return [_to_jsonable(item, on_value, _key_path) for item in obj]
    if type(obj).__name__ == "Command":
        # A LangGraph Command carries the state update a tool applied -- which is the whole payload of
        # DeepAgents' `task` tool, including any files the subagent wrote. Falling through to str() below
        # would record it as an opaque blob. Duck-typed on the class name because this package depends on
        # langchain-core alone and must not import langgraph.
        command = {
            field: _to_jsonable(getattr(obj, field), on_value, (*_key_path, field))
            for field in ("update", "goto", "graph", "resume")
            if getattr(obj, field, None) is not None
        }
        if command:
            return {"command": command}
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump(), on_value, _key_path)
        except Exception:  # noqa: BLE001 - best-effort serialization
            pass
    return str(obj)


# tool name -> Python source captured by @eqty_tool, read by every EqtyCallbackHandler instance
_registered_tool_sources: Dict[str, str] = {}


def eqty_tool(obj: Any) -> Any:
    """Capture a tool function's source code so ``EqtyCallbackHandler`` registers the Tool asset from it.

    Callbacks only receive a tool's *name* at runtime, so the source must be recorded at definition time.
    Returns ``obj`` unchanged and works on either side of LangChain's ``@tool`` decorator::

        @tool
        @eqty_tool
        def search(query: str) -> str: ...
    """
    # when stacked outside @tool, obj is a StructuredTool holding the original fn in .func/.coroutine
    fn = getattr(obj, "func", None) or getattr(obj, "coroutine", None) or obj
    name = getattr(obj, "name", None) or getattr(fn, "__name__", None)
    if name is not None:
        try:
            _registered_tool_sources[name] = inspect.getsource(fn)
        except (OSError, TypeError):
            logger.debug("no source available for tool '%s'", name)
    return obj


class AssetSink:
    """Where an extractor records the assets it produced, and why each one is there.

    *Carried* means the entity already existed and this computation merely handled it, so it is an input.
    *Created* means this computation produced it, so it is an output. Getting the distinction wrong is how
    a lineage graph grows a cycle.
    """

    def __init__(self, metadata: Dict[str, Any]) -> None:
        #: sanitized verbose metadata, ready to unpack into an SDK asset constructor
        self.metadata = metadata
        self.carried: List[CID] = []
        self.created: List[CID] = []

    def carry(self, cid: CID) -> None:
        if cid not in self.carried and cid not in self.created:
            self.carried.append(cid)

    def create(self, cid: CID) -> None:
        if cid not in self.created and cid not in self.carried:
            self.created.append(cid)


class StateExtractor:
    """Lifts part of a graph state into assets of its own, and out of the bulk state blob.

    Without this, everything a node's state contains is re-serialized into that node's state Dataset,
    every time -- so a filesystem carried in state is embedded once per node, and no file is ever an
    entity in its own right. An extractor claims a value, registers whatever assets represent it, and
    returns what should stand in its place in the payload.

    Subclass and register with ``EqtyCallbackHandler.add_extractor`` to teach the handler about a
    framework's own state. The DeepAgents package uses this to turn its virtual filesystem into real
    file assets without this package ever importing ``deepagents``.
    """

    def extract(self, key_path: Tuple[str, ...], value: Any, sink: AssetSink) -> Any:
        """Claim ``value`` and return its replacement, or ``UNCLAIMED`` to decline.

        ``key_path`` is the sequence of dict keys that led here, so an extractor can claim a particular
        state key (``("files",)``) rather than guessing from the value's shape.
        """
        raise NotImplementedError


class PathExtractor(StateExtractor):
    """Registers an existing ``pathlib.Path`` in state as a Dataset of its own.

    Versions are keyed on ``(path, content CID)``, not on the path alone. A file rewritten between two
    nodes is a genuinely different entity, and reusing the first sighting's asset would attest content the
    later computation never saw. The version a rewrite replaced is carried as an input, which is what
    makes successive edits a chain rather than unrelated assets.

    The cost is re-hashing each path per sighting, which is what ``Dataset.from_path`` would do anyway on
    a miss.
    """

    def __init__(self, handler: "EqtyCallbackHandler") -> None:
        self._handler = handler

    def extract(self, key_path: Tuple[str, ...], value: Any, sink: AssetSink) -> Any:
        if not isinstance(value, Path):
            return UNCLAIMED
        if not value.exists():
            return str(value)

        key = str(value.resolve())
        try:
            content_cid = str(get_cid_for_path(value))
        except Exception:  # noqa: BLE001 - an unreadable path must not take down the run being observed
            logger.debug("could not compute a content CID for '%s'; skipping", value)
            return str(value)

        known = self._handler._path_versions.get((key, content_cid))
        if known is not None:
            sink.carry(known)
            return str(value)

        asset = Dataset.from_path(
            value,
            name=value.name,
            description=f"Filesystem asset referenced by LangGraph state: '{value}'.",
            **sink.metadata,
        )
        self._handler._path_versions[(key, content_cid)] = asset.cid

        previous = self._handler._path_latest.get(key)
        if previous is not None:
            sink.carry(previous)
        self._handler._path_latest[key] = asset.cid

        sink.create(asset.cid)
        return str(value)


def _synchronized(method: Callable) -> Callable:
    """Serialize a callback against the handler's lock.

    LangGraph runs the nodes of one superstep concurrently -- two tool calls in a single AI message
    become two ``tools`` tasks on a thread pool -- and it does so in plain synchronous ``.invoke()``,
    not only under ``ainvoke``. Every callback therefore has to assume it may be entered from several
    threads at once, so all of them take the same reentrant lock.
    """

    @functools.wraps(method)
    def wrapper(self: "EqtyCallbackHandler", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class EqtyCallbackHandler(BaseCallbackHandler):
    """Registers LangGraph execution as EQTY data assets and computation statements."""

    def __init__(self, verbose: bool = False) -> None:
        # when True, extra metadata is attached to the registered EQTY assets
        self.verbose = verbose
        if self.verbose:
            logger.info("EqtyCallbackHandler verbose node metadata enabled")
        # guards every mutation below; see _synchronized
        self._lock = threading.RLock()
        # run_id -> tracked run info for graph/node/llm/tool runs we register
        self._runs: Dict[UUID, Dict[str, Any]] = {}
        # run_id -> parent_run_id for every chain run, so nested LLM/tool runs can find their enclosing node even
        # through untracked intermediate runs
        self._parents: Dict[UUID, Optional[UUID]] = {}
        # parent run id -> {superstep -> output-state CIDs produced in it}, used to chain nodes to their
        # predecessors. Keyed by step rather than "the last sibling to finish" so that a superstep running
        # several nodes at once feeds all of their outputs into the next one.
        self._sibling_outputs: Dict[Optional[UUID], Dict[int, List[CID]]] = {}
        # parent run id -> synthetic step counter, for runs LangGraph gives no langgraph_step
        self._fallback_steps: Dict[Optional[UUID], int] = {}
        # run_id -> the agent that run belongs to, inherited from its parent when it names none of its own.
        # A nested run that names a *different* agent is a subagent boundary; see _agent_boundary.
        self._agent_names: Dict[UUID, Optional[str]] = {}
        # which harness produced this run, resolved from run metadata rather than assumed
        self._framework: Optional[str] = None
        # tool identity (JSON) -> Tool asset CID, so one tool configuration is registered once
        self._tool_cids: Dict[str, CID] = {}
        # retriever identity (JSON) -> Tool asset CID, so one retriever config is registered once
        self._retriever_cids: Dict[str, CID] = {}
        # (resolved path, content CID) -> Dataset CID. Keyed on the *contents* as well as the path: the same
        # bytes at the same path are one entity, but rewritten bytes are a new version, and keying on the path
        # alone would attest the previous content for every computation that ran after the change.
        self._path_versions: Dict[Tuple[str, str], CID] = {}
        # resolved path -> Dataset CID of its most recent version, so a rewrite can be linked to what it replaced
        self._path_latest: Dict[str, CID] = {}
        # consulted in order for every value in a state; see add_extractor
        self._extractors: List[StateExtractor] = [PathExtractor(self)]

    ##################################################   Helpers   #################################################
    # kwargs the SDK asset constructors claim for themselves; verbose metadata must not shadow them
    _RESERVED_SDK_KWARGS = frozenset({"obj", "path", "name", "description", "_store"})
    # ls_integration values that identify a component rather than the harness running it
    _COMPONENT_INTEGRATIONS = frozenset({"langchain_chat_model"})

    def _verbose_metadata(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize verbose fields into scalar metadata values safe to unpack into SDK asset constructors.

        Returns ``{}`` unless verbose mode is on, so call sites can unconditionally unpack the result. Values are
        run through ``_to_jsonable`` (stringifying UUIDs, Paths, messages, ...) and non-scalars are JSON-encoded.
        Keys that collide with the SDK's own kwargs (``name``, ``description``, ...) are prefixed with ``LC-`` so
        they can never raise "got multiple values for keyword argument" (dash, not underscore: the graph explorer
        camel-cases keys and turns ``_`` into a space). ``None`` values are kept and encoded as the string
        ``"null"`` so a present-but-empty key is distinguishable from an absent one.
        """
        if not self.verbose:
            return {}
        out: Dict[str, Any] = {}
        # verbose mode attaches raw callback kwargs, which is the third path a credential can take
        for key, value in self._redact(fields).items():
            safe_key = key
            while safe_key in self._RESERVED_SDK_KWARGS or safe_key in out:
                safe_key = f"LC-{safe_key}"
            jsonable = _to_jsonable(value)
            out[safe_key] = jsonable if isinstance(jsonable, (str, int, float, bool)) else json.dumps(jsonable)
        return out

    def add_extractor(self, extractor: StateExtractor) -> None:
        """Teach the handler to lift something out of graph state into assets of its own.

        Extractors are consulted in registration order, and the first to claim a value wins, so a
        subclass registering its own takes precedence over the built-in :class:`PathExtractor`.
        """
        with self._lock:
            self._extractors.insert(0, extractor)

    def _register_state(
        self, obj: Any, name: str, description: str, extra: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dataset, List[CID], List[CID]]:
        """Register ``obj`` as a Dataset, after the extractors have taken what they own.

        Returns ``(state_asset, carried_cids, created_cids)``. Carried assets already existed and are
        linked as inputs; created ones were produced here and become outputs. Never both: re-emitting a
        carried asset as an output puts a cycle in the graph.
        """
        sink = AssetSink(self._verbose_metadata(extra or {}))

        if obj is None:
            # LangChain middleware nodes return None to mean "no state update", and the SDK cannot hash
            # None. Recorded explicitly and scoped to the node that produced it: the node did run, and a
            # single shared empty sentinel would make unrelated nodes converge on one entity in the graph.
            obj = {"state_update": None, "produced_by": name}

        def dispatch(key_path: Tuple[str, ...], value: Any) -> Any:
            for extractor in self._extractors:
                try:
                    claimed = extractor.extract(key_path, value, sink)
                except Exception:  # noqa: BLE001 - a broken extractor must not take down the run
                    logger.debug("extractor %s failed at %s", type(extractor).__name__, key_path)
                    continue
                if claimed is not UNCLAIMED:
                    return claimed
            return UNCLAIMED

        payload = _to_jsonable(obj, on_value=dispatch)
        asset = Dataset.from_object(payload, name=name, description=description, **sink.metadata)

        return asset, sink.carried, sink.created

    #: metadata the frameworks stamp on every run; scoped to one invocation, so including it in an asset
    #: payload would mint a new asset on each call rather than identifying the thing being invoked
    _RUN_SCOPED_METADATA_PREFIXES = ("langgraph_", "ls_", "lc_")
    _RUN_SCOPED_METADATA_KEYS = frozenset({"checkpoint_ns", "thread_id", "run_id", "run_name"})

    #: invocation_params entries that do not describe how the model was asked to sample. The bound tool
    #: belt is the agent's shape rather than the model's -- and each of those tools is already registered
    #: as its own asset, so folding their schemas in here would bloat the Model asset and change it every
    #: time the tool belt does.
    _NON_SAMPLING_PARAMS = frozenset({"tools", "functions", "model", "model_name", "_type"})
    #: Words that mark a parameter as credential material. Matched as whole words, because
    #: `max_completion_tokens` is a sampling knob and not a credential -- "tokens" is not "token".
    _SECRET_PARAM_WORDS = frozenset({"key", "apikey", "token", "secret", "password", "passwd", "pwd", "auth", "bearer"})
    #: Stems for which no ordinary parameter shares the prefix, so a prefix match is safe and catches the
    #: inflections whole-word matching misses -- `authorization` is the standard header name for a
    #: credential, and is not the word "auth". Deliberately excludes anything that would swallow "author".
    _SECRET_PARAM_STEMS = ("secret", "password", "credential", "authoriz", "apikey", "privatekey")
    #: tags the frameworks generate themselves. `seq:step:N` and `graph:step:N` encode a position in a
    #: sequence or a superstep, so treating them as caller intent would mint a new asset for the same
    #: thing invoked at a different point in the graph.
    _FRAMEWORK_TAG_PREFIXES = ("seq:step:", "graph:step:", "map:key:", "langsmith:")

    #: stands in for a redacted value, so a manifest records that a credential was configured without
    #: recording the credential. Constant on purpose: two runs differing only in their key are the same
    #: computation, and folding the key into the CID would both leak it and split the asset.
    _REDACTED = "[redacted]"

    @classmethod
    def _is_secret_param(cls, key: str) -> bool:
        """Whether a parameter name reads as credential material.

        Two rules, because neither alone is right. Whole-word matching keeps `max_completion_tokens` --
        "tokens" is not "token" -- but misses `authorization`, which is not the word "auth". Prefix matching
        catches that, but only for stems no ordinary parameter shares, so `author` is not mistaken for one.
        """
        words = re.split(r"[^a-z0-9]+", key.lower())
        if any(word in cls._SECRET_PARAM_WORDS for word in words):
            return True
        return any(word.startswith(cls._SECRET_PARAM_STEMS) for word in words)

    @classmethod
    def _redact(cls, value: Any, key: Optional[str] = None) -> Any:
        """Replace credential-named values anywhere inside ``value``.

        Applied to everything that can reach an asset payload, and applied *recursively*: a caller can nest
        configuration arbitrarily -- ``extra_body={"api_key": ...}`` is the obvious one -- so checking only
        the keys at the top of a mapping lets the interesting cases through. Asset payloads are stored as
        blobs, so a leak here is a credential written to disk and content-addressed.
        """
        if key is not None and cls._is_secret_param(key):
            return cls._REDACTED
        if isinstance(value, dict):
            return {k: cls._redact(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._redact(item) for item in value]
        return value

    @classmethod
    def _caller_tags(cls, tags: Optional[List[str]]) -> List[str]:
        """The tags the caller attached, with the frameworks' own positional ones removed."""
        return sorted(str(tag) for tag in (tags or []) if not str(tag).startswith(cls._FRAMEWORK_TAG_PREFIXES))

    @classmethod
    def _sampling_params(cls, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The parts of ``invocation_params`` that describe how the model was asked to sample.

        Temperature and the rest change what the model does, so two calls that differ only in sampling are
        genuinely different computations and must not share a Model asset.
        """
        # redacted rather than dropped: a nested `extra_body` may hold both a credential and real
        # configuration, so the structure has to survive with the secret removed from inside it
        return cls._redact({key: value for key, value in (params or {}).items() if key not in cls._NON_SAMPLING_PARAMS})

    @classmethod
    def _caller_metadata(cls, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The metadata the caller attached, with the frameworks' own run-scoped keys removed."""
        return cls._redact(
            {
                key: value
                for key, value in (metadata or {}).items()
                if not key.startswith(cls._RUN_SCOPED_METADATA_PREFIXES) and key not in cls._RUN_SCOPED_METADATA_KEYS
            }
        )

    def _retriever_identity(
        self, name: str, metadata: Optional[Dict[str, Any]], tags: Optional[List[str]]
    ) -> Dict[str, Any]:
        """What identifies this retriever, as opposed to what it returned.

        A retriever is a class pointed at a corpus, and the class name alone cannot tell two of them
        apart -- the same wrapper aimed at a different index looks identical. Whatever the caller attached
        via ``with_config`` is therefore folded in, so a manifest records *which* corpus was consulted and
        not merely that something was.

        ``ls_retriever_name`` is the class, stable even when ``run_name`` renames the run; ``name`` is what
        this run called it. Both are kept rather than guessing which is which.
        """
        identity: Dict[str, Any] = {"retriever": name}

        class_name = (metadata or {}).get("ls_retriever_name")
        if class_name:
            identity["class"] = str(class_name)

        config = self._caller_metadata(metadata)
        if config:
            identity["config"] = _to_jsonable(config)
        retriever_tags = self._caller_tags(tags)
        if retriever_tags:
            identity["tags"] = retriever_tags

        return identity

    def _note_framework(self, metadata: Optional[Dict[str, Any]]) -> None:
        """Record which harness produced this run, the first time a run says so.

        ``ls_integration`` is LangSmith's tag; LangChain's ``create_agent`` sets it to
        ``langchain_create_agent`` and DeepAgents to ``deepagents``. A plain ``StateGraph`` sets neither, so
        the ``langgraph_*`` keys stand in as the evidence there. Anything else is plain LangChain.

        Not every ``ls_integration`` names a harness: langchain-core stamps ``langchain_chat_model`` on
        every model run, which says what the *component* is, not what is orchestrating it. Taking it would
        label a plain LCEL chain after the model it happens to call.
        """
        if self._framework is not None:
            return
        meta = metadata or {}
        integration = meta.get("ls_integration")
        if isinstance(integration, str) and integration and integration not in self._COMPONENT_INTEGRATIONS:
            self._framework = integration
        elif any(key.startswith("langgraph_") for key in meta):
            self._framework = "langgraph"

    @staticmethod
    def _model_identity(
        serialized: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Work out what model this was, from whichever source actually names it.

        ``invocation_params`` is the richest source when a provider fills it in, but nothing requires one
        to: ``GenericFakeChatModel`` reports only ``_type``, and providers vary in whether they use
        ``model`` or ``model_name``. LangSmith's ``ls_model_name`` and ``ls_provider`` are the
        standardised fields and are set from ``_get_ls_params``, so they are the next best thing, and the
        runnable's own class name beats calling a model that plainly exists "unknown".
        """
        meta = metadata or {}
        name = (
            params.get("model")
            or params.get("model_name")
            or meta.get("ls_model_name")
            or (serialized or {}).get("name")
            or "unknown-model"
        )
        provider = meta.get("ls_provider") or params.get("_type") or "unknown"
        return str(name), str(provider)

    def _finalize(self, name: str, kind: str, input_cids: List[CID], output_cids: List[CID]) -> None:
        """Create the computation node w/ metadata."""

        statement_ids = add_computation_statement(inputs=input_cids, outputs=output_cids)

        Metadata(name=name, computation_type=kind, framework=self._framework or "langchain").create_statement(
            statement_ids[0], None, None
        )

    def _finalize_error(self, run: Dict[str, Any], error: BaseException) -> Optional[CID]:
        """Record a failed activity instead of erasing it.

        A tool that raised is part of what happened, and a lineage graph that quietly omits every failure
        overstates how cleanly the run went.
        """
        try:
            failure = Dataset.from_object(
                {"error": type(error).__name__, "message": str(error)},
                name=f"{run['name']}: error",
                description=f"Failure raised by '{run['name']}'.",
            )
        except Exception:  # noqa: BLE001 - never let the observer take down the run it observes
            logger.debug("could not register the failure of '%s'", run.get("name"))
            return None
        self._finalize(run["name"], f"{run['kind']}_error", run.get("inputs", []), [failure.cid])
        return failure.cid

    def _step_for(self, parent_run_id: Optional[UUID], metadata: Optional[Dict[str, Any]]) -> int:
        """The superstep this node belongs to.

        LangGraph numbers supersteps in ``langgraph_step`` and gives every node it runs in parallel the same
        number, which is exactly the ordering the lineage needs. Anything without one -- a plain runnable, a
        non-LangGraph chain -- falls back to a per-parent counter, which reproduces the old sequential
        behaviour for the sequential case.
        """
        step = (metadata or {}).get("langgraph_step")
        if isinstance(step, int) and not isinstance(step, bool):
            return step
        nxt = self._fallback_steps.get(parent_run_id, 0) + 1
        self._fallback_steps[parent_run_id] = nxt
        return nxt

    def _predecessor_outputs(self, parent_run_id: Optional[UUID], step: int) -> List[CID]:
        """Output states of every sibling in the latest superstep that finished before ``step``.

        Returning the whole superstep rather than a single "previous sibling" is what keeps fan-in intact: when
        one model turn issues two tool calls, both ``tools`` nodes run in the same step and the next node is
        derived from both of them.
        """
        by_step = self._sibling_outputs.get(parent_run_id)
        if not by_step:
            return []
        earlier = [s for s in by_step if s < step]
        if not earlier:
            return []
        return list(by_step[max(earlier)])

    def _record_sibling_output(self, parent_run_id: Optional[UUID], step: int, cid: CID) -> None:
        self._sibling_outputs.setdefault(parent_run_id, {}).setdefault(step, []).append(cid)

    def _forget_run(self, run_id: UUID) -> None:
        """Drop the per-parent bookkeeping a finished run owned, so a long session does not accumulate it."""
        self._sibling_outputs.pop(run_id, None)
        self._fallback_steps.pop(run_id, None)
        self._agent_names.pop(run_id, None)

    def _agent_boundary(
        self, run_id: UUID, parent_run_id: Optional[UUID], metadata: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Return this run's agent name if it starts a *new* agent, else None.

        This is LangChain's own rule, from ``langchain.agents._subagent_transformer``: a subagent boundary
        is a nested run whose ``lc_agent_name`` is set and differs from its parent's. Plain subgraphs
        inherit the parent's name and are excluded, which is what keeps every internal LangGraph run from
        being mistaken for an agent.

        Without this a DeepAgents subagent is invisible: its root run carries the subagent's name but
        inherits ``langgraph_node`` from the ``task`` tool that spawned it, so it matches neither the node
        rule nor the root rule, and its internal work is never linked to the call that asked for it.
        """
        own = (metadata or {}).get("lc_agent_name")
        inherited = self._agent_names.get(parent_run_id) if parent_run_id is not None else None
        self._agent_names[run_id] = own or inherited

        if not isinstance(own, str) or not own:
            return None
        if parent_run_id is None:
            return None  # the trace root is handled as the graph itself
        return own if own != inherited else None

    def _enclosing_node(self, parent_run_id: Optional[UUID]) -> Optional[Dict[str, Any]]:
        """Walk up the run tree to the nearest tracked node (or graph) run to get a node so we can link the graph."""

        seen = set()
        current = parent_run_id

        while current is not None and current not in seen:
            seen.add(current)
            run = self._runs.get(current)

            if run is not None:
                return run

            current = self._parents.get(current)

        return None

    ##################################################   Helpers   #################################################

    ################################################## Chain Calls #################################################
    @_synchronized
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
        logger.debug(run_id)
        self._note_framework(metadata)
        self._parents[run_id] = parent_run_id

        # LangGraph emits many internal chain runs (channel reads/writes, task wrappers).
        # A node-level run is the one whose run name equals the "langgraph_node" metadata entry;
        # the root run is the graph itself.
        node = (metadata or {}).get("langgraph_node")
        name = kwargs.get("name") or (serialized or {}).get("name")
        is_node = node is not None and name == node
        is_graph = parent_run_id is None
        subagent = self._agent_boundary(run_id, parent_run_id, metadata)

        if not (is_node or is_graph or subagent):
            logger.debug("untracked chain run %s (%s)", run_id, name)
            return

        # a subagent names itself; an unnamed graph reports itself as "LangGraph", and lc_agent_name is
        # the agent's own name when it has one
        if subagent:
            label = subagent
        elif is_node:
            label = node
        else:
            label = (metadata or {}).get("lc_agent_name") or name or "graph"

        state_in, carried, created = self._register_state(
            inputs,
            name=f"{label}: input state",
            description=f"LangGraph state entering '{label}'.",
            extra={
                "callback": "on_chain_start",
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "metadata": metadata,
                **kwargs,
            },
        )

        # every path in the input state is an input, whether first-seen or not
        input_cids = [state_in.cid, *carried, *created]

        # chain this node to whatever ran in the superstep before it -- possibly several nodes at once
        step = self._step_for(parent_run_id, metadata)

        for prev in self._predecessor_outputs(parent_run_id, step):
            if prev != state_in.cid and prev not in input_cids:
                input_cids.append(prev)

        if subagent:
            kind = "agent"
        elif is_graph:
            kind = "graph"
        else:
            kind = "graph_node"

        self._runs[run_id] = {
            "name": label,
            "kind": kind,
            "parent": parent_run_id,
            "step": step,
            "state_in": state_in.cid,
            "inputs": input_cids,
            "child_outputs": [],
        }

    @_synchronized
    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        logger.debug(run_id)
        self._parents.pop(run_id, None)
        run = self._runs.pop(run_id, None)

        if run is None:
            return

        label = run["name"]

        state_out, carried, created = self._register_state(
            outputs,
            name=f"{label}: output state",
            description=f"LangGraph state produced by '{label}'.",
            extra={"callback": "on_chain_end", "run_id": run_id, **kwargs},
        )

        # nested LLM/tool outputs are inputs to the node's final state
        input_cids = run["inputs"] + run["child_outputs"]

        if run["kind"] in ("graph", "agent"):
            # the final state is derived from the last node that ran inside it
            last = run.get("last_child_output")
            if last is not None and last not in input_cids:
                input_cids.append(last)

        # a path first seen in this output state was created here -> output;
        # a path registered earlier is only carried through -> input
        input_cids += [c for c in carried if c not in input_cids]
        output_cids = [state_out.cid, *created]
        self._finalize(label, run["kind"], input_cids, output_cids)

        if run["parent"] is not None:
            self._record_sibling_output(run["parent"], run["step"], state_out.cid)
        else:
            # A root run has no siblings, so its output has nowhere to be recorded -- and the bucket it
            # would land in is keyed None, which no _forget_run can reach. Clearing it here is what stops
            # a reused handler from chaining the next invocation's root onto this one's output.
            self._sibling_outputs.pop(None, None)
            self._fallback_steps.pop(None, None)

        self._forget_run(run_id)
        enclosing = self._enclosing_node(run["parent"])

        if enclosing is not None:
            enclosing["last_child_output"] = state_out.cid
            if run["kind"] == "agent":
                # the enclosing run is the tool that spawned this subagent -- DeepAgents' `task`. Its
                # result *is* the subagent's final state, so without this edge the whole delegated run
                # hangs off nothing and the graph has a hole exactly where the work was handed over.
                enclosing.setdefault("child_outputs", []).append(state_out.cid)

    @_synchronized
    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        logger.warning("%s run %s failed: %s: %s", "chain", run_id, type(error).__name__, error)
        self._parents.pop(run_id, None)
        run = self._runs.pop(run_id, None)
        self._forget_run(run_id)

        if run is not None:
            self._finalize_error(run, error)

    ################################################## Chain Calls #################################################

    ################################################## LLM Calls ###################################################
    @_synchronized
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
        logger.debug(run_id)
        self._note_framework(metadata)
        params = kwargs.get("invocation_params") or {}
        model_name, provider = self._model_identity(serialized, params, metadata)

        prompt = Prompt.from_object(
            _to_jsonable(messages),
            name=f"{model_name}: prompt",
            description="Messages sent to the chat model.",
            **self._verbose_metadata(
                {
                    "callback": "on_chat_model_start",
                    "serialized": serialized,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "metadata": metadata,
                    **kwargs,
                }
            ),
        )

        model_payload: Dict[str, Any] = {"model": model_name, "provider": provider}
        sampling = self._sampling_params(params)
        if sampling:
            model_payload["params"] = _to_jsonable(sampling)
        model_config = self._caller_metadata(metadata)
        if model_config:
            model_payload["config"] = _to_jsonable(model_config)
        model_tags = self._caller_tags(tags)
        if model_tags:
            model_payload["tags"] = model_tags

        model = Model.from_object(
            model_payload,
            name=model_name,
            **self._verbose_metadata(
                {
                    "callback": "on_chat_model_start",
                    "serialized": serialized,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "metadata": metadata,
                    **kwargs,
                }
            ),
        )

        input_cids = [prompt.cid, model.cid]
        node = self._enclosing_node(parent_run_id)
        if node is not None:
            # the prompt is derived from whatever the enclosing activity started from -- a node's input
            # state, or a tool's arguments when the model is being called from inside a tool
            enclosing_input = node.get("state_in")
            if enclosing_input is not None:
                input_cids.append(enclosing_input)

        self._runs[run_id] = {
            "name": model_name,
            "kind": "chat_model",
            "state_in": prompt.cid,
            "inputs": input_cids,
            "child_outputs": [],
            "node": node,
        }

    @_synchronized
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        logger.debug(run_id)
        run = self._runs.pop(run_id, None)

        if run is None:
            return

        generations = [
            _to_jsonable(getattr(gen, "message", None) or gen.text) for batch in response.generations for gen in batch
        ]

        output = Reasoning.from_object(
            generations,
            name=f"{run['name']}: response",
            description="Chat model response, including any tool calls.",
            **self._verbose_metadata({"callback": "on_llm_end", "run_id": run_id, **kwargs}),
        )

        self._finalize(run["name"], run["kind"], run["inputs"], [output.cid])

        if run["node"] is not None:
            run["node"].setdefault("child_outputs", []).append(output.cid)

    @_synchronized
    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        logger.warning("%s run %s failed: %s: %s", "llm", run_id, type(error).__name__, error)
        run = self._runs.pop(run_id, None)

        if run is not None:
            self._finalize_error(run, error)

    ################################################## LLM Calls ###################################################

    ################################################## Tool Calls ##################################################
    @_synchronized
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
        logger.debug(run_id)
        tool_name = (serialized or {}).get("name", "tool")

        # an @eqty_tool-decorated tool is registered from its source code, like @compute's code asset;
        # an undecorated tool falls back to a name/description stub. Either way, configuration the caller
        # attached with `with_config` identifies *this* tool as opposed to another of the same name, so it
        # is folded in -- and the payload becomes a mapping, since a bare source string has nowhere to put
        # it. A tool with nothing attached keeps the source-as-payload shape it has always had.
        source = _registered_tool_sources.get(tool_name)
        tool_config = self._caller_metadata(metadata)
        tool_tags = self._caller_tags(tags)

        if tool_config or tool_tags:
            tool_payload: Any = {"tool": tool_name}
            if source is not None:
                tool_payload["source"] = source
            if tool_config:
                tool_payload["config"] = _to_jsonable(tool_config)
            if tool_tags:
                tool_payload["tags"] = tool_tags
        else:
            tool_payload = source if source is not None else {"name": tool_name}

        # keyed on the payload, so one name configured two ways registers as two assets
        tool_key = json.dumps(tool_payload, sort_keys=True, default=str)

        if tool_key not in self._tool_cids:
            tool_asset = Tool.from_object(
                tool_payload,
                name=tool_name,
                description=(serialized or {}).get("description", ""),
                **self._verbose_metadata(
                    {
                        "callback": "on_tool_start",
                        "run_id": run_id,
                        "parent_run_id": parent_run_id,
                        "tags": tags,
                        "metadata": metadata,
                        **kwargs,
                    }
                ),
            )
            self._tool_cids[tool_key] = tool_asset.cid

        tool_input, carried, created = self._register_state(
            inputs if inputs is not None else input_str,
            name=f"{tool_name}: input",
            description=f"Arguments passed to tool '{tool_name}'.",
            extra={
                "callback": "on_tool_start",
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "metadata": metadata,
                **kwargs,
            },
        )

        input_cids = [self._tool_cids[tool_key], tool_input.cid, *carried, *created]

        node = self._enclosing_node(parent_run_id)
        if node is not None:
            # the tool call was requested by the state entering the node
            enclosing_input = node.get("state_in")
            if enclosing_input is not None:
                input_cids.append(enclosing_input)

        self._runs[run_id] = {
            "name": tool_name,
            "kind": "tool",
            # a tool's arguments are what anything nested inside it derives from; keeping the key uniform
            # across run kinds is what lets _enclosing_node return any of them
            "state_in": tool_input.cid,
            "inputs": input_cids,
            "child_outputs": [],
            "node": node,
        }

    @_synchronized
    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        logger.debug(run_id)
        run = self._runs.pop(run_id, None)

        if run is None:
            return

        tool_output, carried, created = self._register_state(
            output,
            name=f"{run['name']}: output",
            description=f"Result returned by tool '{run['name']}'.",
            extra={"callback": "on_tool_end", "run_id": run_id, **kwargs},
        )

        # whatever ran inside the tool -- a nested model call, a nested chain -- contributed to its result,
        # the same way a node's children contribute to the node's output state
        input_cids = run["inputs"] + run.get("child_outputs", [])
        input_cids += [c for c in carried if c not in input_cids]
        output_cids = [tool_output.cid, *created]
        self._finalize(run["name"], run["kind"], input_cids, output_cids)

        if run["node"] is not None:
            run["node"].setdefault("child_outputs", []).extend(output_cids)

    @_synchronized
    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        logger.warning("%s run %s failed: %s: %s", "tool", run_id, type(error).__name__, error)
        run = self._runs.pop(run_id, None)

        if run is None:
            return

        failure = self._finalize_error(run, error)

        # the error message goes back to the model as a ToolMessage, so the enclosing node's output state
        # is derived from the failure just as it would be from a successful result
        if failure is not None and run["node"] is not None:
            run["node"].setdefault("child_outputs", []).append(failure)

    ################################################## Tool Calls ##################################################

    ################################################## Retrievers ##################################################
    @_synchronized
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Register a retrieval as its own computation.

        For a RAG chain the retrieved documents are the provenance that matters most -- they are where
        the answer's content actually came from -- and they were leaving no trace at all. The retriever
        itself is registered as a Tool, because that is what it is from the graph's point of view: a named
        capability the run invoked.

        Its asset is content-addressed to its identity, which includes whatever the caller attached with
        ``with_config`` -- so pointing the same class at a different index produces a different asset,
        and the manifest records which corpus was consulted rather than only that one was.
        """
        logger.debug(run_id)
        self._note_framework(metadata)
        name = kwargs.get("name") or (serialized or {}).get("name") or "retriever"
        identity = self._retriever_identity(name, metadata, tags)
        # keyed on the identity rather than the name, so the same class aimed at two different corpora
        # registers as two assets instead of silently collapsing into one
        key = json.dumps(identity, sort_keys=True)

        if key not in self._retriever_cids:
            asset = Tool.from_object(
                identity,
                name=name,
                description=(serialized or {}).get("description", ""),
                **self._verbose_metadata(
                    {
                        "callback": "on_retriever_start",
                        "serialized": serialized,
                        "run_id": run_id,
                        "parent_run_id": parent_run_id,
                        "tags": tags,
                        "metadata": metadata,
                        **kwargs,
                    }
                ),
            )
            self._retriever_cids[key] = asset.cid

        query_asset = Prompt.from_object(
            _to_jsonable(query),
            name=f"{name}: query",
            description=f"Query issued to retriever '{name}'.",
            **self._verbose_metadata(
                {
                    "callback": "on_retriever_start",
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "metadata": metadata,
                    **kwargs,
                }
            ),
        )

        input_cids = [self._retriever_cids[key], query_asset.cid]
        node = self._enclosing_node(parent_run_id)
        if node is not None:
            enclosing_input = node.get("state_in")
            if enclosing_input is not None:
                input_cids.append(enclosing_input)

        self._runs[run_id] = {
            "name": name,
            "kind": "retriever",
            "state_in": query_asset.cid,
            "inputs": input_cids,
            "child_outputs": [],
            "node": node,
        }

    @_synchronized
    def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Each retrieved document becomes its own Document asset.

        One asset per document rather than one per result set: the same document retrieved by two
        different queries is the same entity, and content-addressing makes that identity automatic. A
        single blob of "the results" would hide it.
        """
        logger.debug(run_id)
        run = self._runs.pop(run_id, None)

        if run is None:
            return

        output_cids: List[CID] = []
        for index, document in enumerate(documents or []):
            payload = _to_jsonable(document)
            source = None
            if isinstance(payload, dict):
                source = (payload.get("metadata") or {}).get("source")
            asset = Document.from_object(
                payload,
                name=str(source) if source else f"{run['name']}: document {index + 1}",
                description=f"Document retrieved by '{run['name']}'.",
                **self._verbose_metadata({"callback": "on_retriever_end", "run_id": run_id, **kwargs}),
            )
            if asset.cid not in output_cids:
                output_cids.append(asset.cid)

        self._finalize(run["name"], run["kind"], run["inputs"], output_cids)

        # the documents are what the enclosing node's output was built from
        if run["node"] is not None:
            run["node"].setdefault("child_outputs", []).extend(output_cids)

    @_synchronized
    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        logger.warning("retriever run %s failed: %s: %s", run_id, type(error).__name__, error)
        run = self._runs.pop(run_id, None)

        if run is not None:
            self._finalize_error(run, error)

    ################################################## Retrievers ##################################################


__all__ = [
    "UNCLAIMED",
    "AssetSink",
    "EqtyCallbackHandler",
    "PathExtractor",
    "StateExtractor",
    "eqty_tool",
]
