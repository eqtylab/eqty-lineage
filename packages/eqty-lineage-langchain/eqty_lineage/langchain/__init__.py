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

The module is split by concern: ``_serialize`` turns LangChain values into plain JSON, ``_tools`` captures tool
source at definition time, ``extractors`` holds the extension point, and this file holds the handler itself.
"""

import functools
import json
import logging
import re
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from eqty_sdk import CID, Context, Dataset, Document, Model, Prompt, Reasoning, Service, Tool, init
from eqty_sdk.metadata import Metadata
from eqty_sdk.statements import add_computation_statement

from eqty_lineage.langchain._serialize import UNCLAIMED, _to_jsonable
from eqty_lineage.langchain._tools import _registered_tool_sources, eqty_tool
from eqty_lineage.langchain.extractors import AssetSink, PathExtractor, StateExtractor

logger = logging.getLogger("eqty.langgraph")


def _is_graph_control_flow(error: BaseException) -> bool:
    """Whether LangGraph raised this to move the graph rather than to report a failure.

    ``GraphInterrupt`` (a pause for approval), ``ParentCommand`` and ``GraphDelegate`` all subclass
    ``GraphBubbleUp`` and all reach ``on_chain_error`` looking like a crash. Matched on the base class
    *name*: this package depends on ``langchain-core`` alone and must not import ``langgraph``.
    """
    return any(cls.__name__ == "GraphBubbleUp" for cls in type(error).__mro__)


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


def _log_run_ended(kind: str, run_id: UUID, error: BaseException) -> None:
    """One place deciding how an ended run is announced, so a pause never reads as a crash in the logs."""
    if _is_graph_control_flow(error):
        logger.debug("%s run %s suspended: %s", kind, run_id, type(error).__name__)
    else:
        logger.warning("%s run %s failed: %s: %s", kind, run_id, type(error).__name__, error)


class EqtyCallbackHandler(BaseCallbackHandler):
    """Registers LangGraph execution as EQTY data assets and computation statements."""

    # Contexts are process-wide SDK objects, as is ``eqty_sdk.init``. The cache lets a new handler for a
    # later turn of the same LangGraph thread use the same child context.
    _thread_contexts: Dict[Tuple[str, str], Context] = {}
    _thread_context_lock = threading.RLock()

    def __init__(
        self,
        verbose: bool = False,
        *,
        integrity_service_url: Optional[str] = None,
    ) -> None:
        # when True, extra metadata is attached to the registered EQTY assets
        self.verbose = verbose
        # The application selects this root once with ``eqty_sdk.init(default_context=...)``. LangGraph
        # ``thread_id`` values are mapped to child contexts beneath it as callbacks begin.
        self._root_context = init().get_default_context()
        self._context = self._root_context
        # Registration is opt-in. Service.new resolves its credentials exclusively from EQTY_API_KEY.
        self._service = Service.new(integrity_service_url) if integrity_service_url else None
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

    def _asset_factory(self, asset_type: Any) -> Any:
        """Return an SDK asset factory explicitly bound to this handler's context."""
        return asset_type.with_context(self._context) if self._context is not None else asset_type

    @property
    def context(self) -> Context:
        """The EQTY root or LangGraph-thread child context used by this handler's latest invocation."""
        return self._context

    def _register_context(self, context: Optional[Context] = None) -> None:
        """Push one graph invocation's context, when service registration is configured.

        A LangGraph ``thread_id`` can span many user messages.  Registering here, rather than
        waiting for that thread/session to end, makes the lineage from each ``invoke`` available
        to Integrity Service immediately.
        """
        if self._service is None:
            return
        registered_context = context if context is not None else self._context
        logger.info(
            "integrity_service.registering context_id=%s context_name=%s",
            registered_context.id,
            registered_context.name,
        )
        registered_context.register(self._service)
        logger.info(
            "integrity_service.registered context_id=%s context_name=%s",
            registered_context.id,
            registered_context.name,
        )

    def _activate_thread_context(self, metadata: Optional[Dict[str, Any]], agent_name: Optional[Any] = None) -> None:
        """Select the child context for this LangGraph thread, or the configured root without one."""
        thread_id = (metadata or {}).get("thread_id")
        if thread_id is None:
            # Nested LangChain callbacks do not always propagate LangGraph's configurable metadata.
            # Keep the context selected by their enclosing graph rather than falling back to root.
            return

        key = (str(self._root_context.id), str(thread_id))
        with self._thread_context_lock:
            context = self._thread_contexts.get(key)
            if context is None:
                name = str(agent_name or (metadata or {}).get("lc_agent_name") or "LangGraph")
                timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                context = Context.with_parent(self._root_context).new(f"{name}: {timestamp}")
                self._thread_contexts[key] = context
        self._context = context

    def _bind_context_to_root_run(self, parent_run_id: Optional[UUID]) -> None:
        """Update the enclosing root run when a nested callback first reveals its thread context."""
        current = parent_run_id
        seen = set()
        while current is not None and current not in seen:
            seen.add(current)
            run = self._runs.get(current)
            if run is not None and run["parent"] is None:
                run["context"] = self._context
                return
            current = self._parents.get(current)

    def _verbose_metadata(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize verbose fields into scalars safe to unpack into SDK asset constructors.

        Returns ``{}`` unless verbose mode is on, so call sites can unpack unconditionally. Keys colliding
        with the SDK's own kwargs get an ``LC-`` prefix; ``None`` is encoded as ``"null"`` so a
        present-but-empty key stays distinguishable from an absent one.
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

        Extractors are consulted newest-first -- each is inserted at the front -- and the first to claim a
        value wins, so a subclass's own extractor takes precedence over the built-in :class:`PathExtractor`.
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
        asset = self._asset_factory(Dataset).from_object(payload, name=name, description=description, **sink.metadata)

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

        The class name alone cannot tell two retrievers apart -- the same wrapper aimed at a different
        index looks identical -- so whatever the caller attached via ``with_config`` is folded in.
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

        ``ls_integration`` names it (``langchain_create_agent``, ``deepagents``); a plain ``StateGraph``
        sets neither, so the ``langgraph_*`` keys stand in. ``langchain_chat_model`` is excluded -- it
        names a component, not a harness, and would label an LCEL chain after the model it calls.
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

        Providers vary between ``model`` and ``model_name`` and some fill in neither, so this falls back
        through ``ls_model_name``/``ls_provider`` to the runnable's class name before giving up.
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

        statement_ids = add_computation_statement(inputs=input_cids, outputs=output_cids, context=self._context)

        Metadata(name=name, computation_type=kind, framework=self._framework or "langchain").create_statement(
            statement_ids[0], None, self._context
        )

    def _record_failure(self, run: Optional[Dict[str, Any]], error: BaseException) -> None:
        """Record a failed activity and link it to whatever was waiting on it.

        Every ``on_*_error`` funnels through here. The caller is linked because it observed the failure
        whether or not it recovered -- and when it dies too, its own error computation is built from its
        inputs and the link goes unused.
        """
        if run is None:
            return
        if _is_graph_control_flow(error):
            # the run did not fail, it suspended; the turn that resumes it records the node normally, so
            # dropping it here loses nothing and inventing a failure would lose the truth
            return
        failure = self._finalize_error(run, error)
        enclosing = run.get("node")
        if failure is not None and enclosing is not None:
            enclosing.setdefault("child_outputs", []).append(failure)

    def _finalize_error(self, run: Dict[str, Any], error: BaseException) -> Optional[CID]:
        """Record a failed activity instead of erasing it.

        A tool that raised is part of what happened, and a lineage graph that quietly omits every failure
        overstates how cleanly the run went.
        """
        try:
            failure = self._asset_factory(Dataset).from_object(
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

        LangChain's own rule, from ``langchain.agents._subagent_transformer``: a nested run whose
        ``lc_agent_name`` is set and differs from its parent's. Plain subgraphs inherit the parent name and
        are excluded, which keeps internal LangGraph runs from being mistaken for agents.
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
        name = kwargs.get("name") or (serialized or {}).get("name")
        self._activate_thread_context(metadata, agent_name=(metadata or {}).get("lc_agent_name") or name)
        self._note_framework(metadata)
        self._parents[run_id] = parent_run_id
        self._bind_context_to_root_run(parent_run_id)

        # LangGraph emits many internal chain runs (channel reads/writes, task wrappers).
        # A node-level run is the one whose run name equals the "langgraph_node" metadata entry;
        # the root run is the graph itself.
        node = (metadata or {}).get("langgraph_node")
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
            # Registration must use the graph's context even if a later nested callback supplies
            # incomplete metadata and changes the handler's active context.
            "context": self._context,
        }

    @_synchronized
    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        logger.debug(run_id)
        self._parents.pop(run_id, None)
        run = self._runs.pop(run_id, None)
        # released before the early return: every chain run gets an _agent_names entry at start, tracked or
        # not, and LangGraph emits far more untracked runs than tracked ones. Releasing this only on the
        # tracked path left the dict growing for the whole session.
        self._forget_run(run_id)

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

        enclosing = self._enclosing_node(run["parent"])

        if enclosing is not None:
            enclosing["last_child_output"] = state_out.cid
            if run["kind"] == "agent":
                # the enclosing run is the tool that spawned this subagent -- DeepAgents' `task`. Its
                # result *is* the subagent's final state, so without this edge the whole delegated run
                # hangs off nothing and the graph has a hole exactly where the work was handed over.
                enclosing.setdefault("child_outputs", []).append(state_out.cid)

        # ``parent is None`` identifies the graph invocation, not the browser/chat session.
        # A thread can make many such calls, and each one is registered independently.
        if run["parent"] is None:
            self._register_context(run["context"])

    @_synchronized
    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        _log_run_ended("chain", run_id, error)
        self._parents.pop(run_id, None)
        run = self._runs.pop(run_id, None)
        self._forget_run(run_id)
        self._record_failure(run, error)
        if run is not None and run["parent"] is None:
            self._register_context(run["context"])

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
        self._activate_thread_context(metadata, agent_name=kwargs.get("name"))
        self._bind_context_to_root_run(parent_run_id)
        self._note_framework(metadata)
        params = kwargs.get("invocation_params") or {}
        model_name, provider = self._model_identity(serialized, params, metadata)

        prompt = self._asset_factory(Prompt).from_object(
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

        model = self._asset_factory(Model).from_object(
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

        output = self._asset_factory(Reasoning).from_object(
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
        _log_run_ended("llm", run_id, error)
        self._record_failure(self._runs.pop(run_id, None), error)

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
        self._activate_thread_context(metadata, agent_name=kwargs.get("name"))
        self._bind_context_to_root_run(parent_run_id)
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
            tool_asset = self._asset_factory(Tool).from_object(
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
        _log_run_ended("tool", run_id, error)
        self._record_failure(self._runs.pop(run_id, None), error)

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

        The retriever is registered as a Tool -- a named capability the run invoked -- content-addressed
        to its identity, so the same class pointed at a different index is a different asset.
        """
        logger.debug(run_id)
        self._activate_thread_context(metadata, agent_name=kwargs.get("name"))
        self._bind_context_to_root_run(parent_run_id)
        self._note_framework(metadata)
        name = kwargs.get("name") or (serialized or {}).get("name") or "retriever"
        identity = self._retriever_identity(name, metadata, tags)
        # keyed on the identity rather than the name, so the same class aimed at two different corpora
        # registers as two assets instead of silently collapsing into one
        key = json.dumps(identity, sort_keys=True)

        if key not in self._retriever_cids:
            asset = self._asset_factory(Tool).from_object(
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

        query_asset = self._asset_factory(Prompt).from_object(
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
            asset = self._asset_factory(Document).from_object(
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
        _log_run_ended("retriever", run_id, error)
        self._record_failure(self._runs.pop(run_id, None), error)

    ################################################## Retrievers ##################################################


__all__ = [
    "UNCLAIMED",
    "AssetSink",
    "EqtyCallbackHandler",
    "PathExtractor",
    "StateExtractor",
    "eqty_tool",
]
