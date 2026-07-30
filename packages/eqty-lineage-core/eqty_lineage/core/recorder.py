"""Turns the agent event vocabulary into EQTY assets, statements, and triples.

The recorder is deliberately the only place that talks to ``eqty_sdk``. Adapters produce
:mod:`~eqty_lineage.core.events` and nothing else, which is what makes the offline and live capture paths
comparable: if they emit the same events they must produce the same graph, and any divergence is a bug in
one adapter rather than a difference of opinion about the SDK.

Two behaviours here differ from the LangChain handler this was generalized from, both because coding
agents break assumptions that hold for a graph run:

*Files are versioned, not identified.* The LangChain handler keys registered paths on the path alone and
treats a second sighting as "carried, not created". For an agent whose whole purpose is read -> edit ->
read again, that rule either hides every edit or cycles the graph. Here the key is
``(path, content CID)``, so the same bytes are the same entity and different bytes are a new version with
a derivation edge to its predecessor.

*Absence is recorded.* Compaction boundaries and opaque subagents are emitted as explicit nodes. A graph
that quietly omits them asserts a completeness it does not have, which is worse than no graph at all.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from eqty_sdk import (
    CID,
    Agent,
    Code,
    Configuration,
    Dataset,
    Document,
    Guardrail,
    Model,
    Prompt,
    Reasoning,
    SystemPrompt,
    Tool,
    get_cid_for_bytes,
)
from eqty_sdk.context import get_active_context
from eqty_sdk.metadata import Metadata
from eqty_sdk.statements import ASSOCIATION_TYPES, Association, add_computation_statement

from . import prov
from .events import (
    Compacted,
    Event,
    FileObserved,
    InstructionsLoaded,
    ModelCall,
    PermissionDecision,
    PromptSubmitted,
    SessionEnded,
    SessionStarted,
    SubagentEnded,
    SubagentStarted,
    ToolCallEnded,
    ToolCallStarted,
)
from .redaction import ContentPolicy
from .serialize import JsonableHook, as_bytes, scalar_metadata, to_jsonable
from .triples import TripleSink

logger = logging.getLogger("eqty.lineage.core")

# Extension -> asset type. The SDK already ships 22 asset types; a coding agent maps onto the existing
# set with no new primitives, which keeps manifests readable in the graph explorer (the type drives the
# icon) and keeps this package from inventing vocabulary EQTY would then have to support.
_CODE_SUFFIXES = frozenset(
    """.py .pyi .js .jsx .ts .tsx .rs .go .java .kt .swift .c .h .cc .cpp .hpp .cs .rb .php .scala
    .clj .ex .exs .erl .hs .lean .ml .mli .sh .bash .zsh .fish .sql .lua .r .jl .dart .vue .svelte
    .proto .tf .nix""".split()
)
_DOC_SUFFIXES = frozenset(".md .markdown .rst .txt .adoc .org".split())
_CONFIG_SUFFIXES = frozenset(".json .yaml .yml .toml .ini .cfg .conf .env .properties".split())


def _asset_class_for(path: str) -> Type[Any]:
    suffix = ("." + path.rsplit(".", 1)[-1].lower()) if "." in path.rsplit("/", 1)[-1] else ""
    if suffix in _CODE_SUFFIXES:
        return Code
    if suffix in _DOC_SUFFIXES:
        return Document
    if suffix in _CONFIG_SUFFIXES:
        return Configuration
    return Dataset


@dataclass
class FileVersion:
    """One content-addressed state of one path."""

    path: str
    content_cid: str
    asset_cid: CID
    version: int
    observed: bool = True
    redacted: bool = False
    user_modified: bool = False


@dataclass
class _Run:
    """An open tool call, accumulating inputs and outputs until its end event arrives."""

    tool_use_id: str
    tool_name: str
    kind: str
    inputs: List[CID] = field(default_factory=list)
    outputs: List[CID] = field(default_factory=list)
    parent_id: Optional[str] = None
    observed: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class LineageRecorder:
    """Consumes agent events; produces EQTY statements and a triple fact set.

    One recorder per session. It is not thread-safe: the HTTP hook daemon must serialize events per
    session, which it gets for free by keying recorders on ``session_id``.
    """

    def __init__(
        self,
        policy: Optional[ContentPolicy] = None,
        triples: Optional[TripleSink] = None,
        framework: str = "agent",
        verbose: bool = False,
        jsonable_hook: Optional[JsonableHook] = None,
    ) -> None:
        self.policy = policy if policy is not None else ContentPolicy()
        self.triples = triples if triples is not None else TripleSink()
        self.framework = framework
        self.verbose = verbose
        self._hook = jsonable_hook

        self._session_id: Optional[str] = None
        self._agent_cid: Optional[CID] = None
        # instructions and configuration that stand as inputs to everything the session subsequently does
        self._session_inputs: List[CID] = []
        # a moving anchor for "the conversation state", so compaction has something to be an edge between
        self._context_anchor: Optional[CID] = None
        self._permission_mode: Optional[str] = None
        self._effort: Optional[str] = None

        self._tool_assets: Dict[str, CID] = {}
        self._runs: Dict[str, _Run] = {}
        self._file_versions: Dict[str, List[FileVersion]] = {}
        self._by_content: Dict[Tuple[str, str], FileVersion] = {}
        self._subagents: Dict[str, Dict[str, Any]] = {}
        # decisions arrive before the call they authorize finishes, so they wait here for its activity CID
        self._pending_permissions: Dict[str, PermissionDecision] = {}
        self._last_activity: Optional[CID] = None
        # The most recent *model call* specifically. `triggered` means "the model asked for this", so it
        # may only ever originate at a model call. Attributing it to whatever merely ran last would let
        # the hook path -- which never sees model calls -- assert that one tool call triggered the next.
        self._last_model_activity: Optional[CID] = None

        self.stats: Dict[str, int] = {}

    # ------------------------------------------------------------------ public API
    def handle(self, event: Event) -> None:
        """Route one event. Unknown event types are ignored, not fatal.

        Adapters run against agents that add hook events between releases; a recorder that crashed on an
        unrecognized event would take down the session it is supposed to be observing.
        """
        handler = self._DISPATCH.get(type(event).__name__)
        if handler is None:
            logger.debug("no handler for %s", type(event).__name__)
            return
        self.stats[type(event).__name__] = self.stats.get(type(event).__name__, 0) + 1
        handler(self, event)

    def handle_all(self, events) -> None:
        for event in events:
            self.handle(event)

    @property
    def file_versions(self) -> Dict[str, List[FileVersion]]:
        return self._file_versions

    # ------------------------------------------------------------------ helpers
    def _meta(self, **fields: Any) -> Dict[str, Any]:
        """Metadata unconditionally attached to assets. Verbose fields are opt-in; these are not."""
        base = {
            prov.K_FRAMEWORK: self.framework,
            prov.K_SESSION: self._session_id,
        }
        base.update(fields)
        return scalar_metadata({k: v for k, v in base.items() if v is not None}, self._hook)

    def _verbose(self, **fields: Any) -> Dict[str, Any]:
        return scalar_metadata(fields, self._hook) if self.verbose else {}

    def _edge(self, subject: Any, predicate: str, obj: Any, observed: bool = True) -> None:
        self.triples.add(subject, predicate, obj, session_id=self._session_id, observed=observed)

    def _finalize(
        self,
        name: str,
        kind: str,
        inputs: List[CID],
        outputs: List[CID],
        observed: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        authorized_by: Optional[CID] = None,
    ) -> Optional[CID]:
        """Create the computation statement plus its metadata, and mirror both into the fact set.

        Returns the activity CID -- the statement itself, which is what ``prov:used`` and
        ``prov:wasGeneratedBy`` hang off and what an ``authorizedBy`` association attaches to.
        """
        inputs = _dedupe(inputs)
        outputs = _dedupe(outputs)
        if not outputs:
            # A computation with no output is not a lineage edge; recording one would create a node
            # nothing can be downstream of and quietly inflate the graph.
            logger.debug("skipping computation '%s' with no outputs", name)
            return None

        # Assets resolve the active context themselves (Asset._from_object falls back to
        # get_active_context), but statement constructors take an explicit kwarg and do *not*. Omitting
        # it sends every computation to the process default context while the assets it references sit
        # in another -- the graph splits in half and the export comes back empty. Resolve it here so
        # both halves land together under `with graph_context(ctx)`.
        ctx = get_active_context()

        statement_ids = add_computation_statement(inputs=inputs, outputs=outputs, context=ctx)
        activity = statement_ids[0]

        meta = {
            "name": name,
            "computation_type": kind,
            prov.K_FRAMEWORK: self.framework,
            prov.K_OBSERVED: observed,
            prov.K_SESSION: self._session_id,
            prov.K_PERMISSION_MODE: self._permission_mode,
            prov.K_EFFORT: self._effort,
        }
        if extra:
            meta.update(extra)
        Metadata(**{k: v for k, v in meta.items() if v is not None}).create_statement(activity, None, ctx)

        self._edge(activity, prov.LABEL, name)
        for cid in inputs:
            self._edge(activity, prov.USED, cid, observed)
        for cid in outputs:
            self._edge(cid, prov.WAS_GENERATED_BY, activity, observed)
        if self._agent_cid is not None:
            self._edge(activity, prov.RAN_AS, self._agent_cid)

        if authorized_by is not None:
            # Association is the SDK's own typed-edge mechanism; CERTIFIES is the closest of the three
            # exposed types to "this decision vouched for this activity".
            try:
                factory = Association.with_context(ctx) if ctx is not None else Association
                factory.new(activity, ASSOCIATION_TYPES.CERTIFIES).add_predicate(authorized_by).finalize()
            except Exception:  # noqa: BLE001 - an association failure must not lose the computation
                logger.warning("could not associate guardrail with activity", exc_info=True)
            self._edge(activity, prov.AUTHORIZED_BY, authorized_by)

        self._last_activity = activity
        return activity

    def _asset(self, asset_cls: Type[Any], payload: Any, **kwargs: Any) -> Any:
        """Create an asset and record its type in the fact set.

        Every asset goes through here so the type triple can never be forgotten on a new call site.
        """
        asset = asset_cls.from_object(payload, **kwargs)
        type_name = getattr(getattr(asset_cls, "_asset_type", None), "value", asset_cls.__name__)
        self._edge(asset.cid, prov.ASSET_TYPE, type_name)
        return asset

    # ------------------------------------------------------------------ file versions
    def observe_file(self, event: FileObserved) -> Optional[FileVersion]:
        """Register one file version, deduplicating on content.

        The same bytes at the same path are the same entity no matter how often they are seen; different
        bytes are a new version carrying a derivation edge back to the one it replaced. That edge is only
        promoted to its own computation statement when no tool call is open to own it -- inside a tool
        run the prev/next pair becomes that run's input/output, which is the same PROV shape without a
        duplicate activity.
        """
        path = event.path
        if not path:
            return None

        deleted = event.mode == "deleted"
        data = as_bytes(event.content) if event.content is not None else None
        # Identity is always the CID of the *original* bytes, computed without storing them. Keying on
        # post-scrub content would collapse two different secrets into one version.
        #
        # A tombstone has no bytes to hash, and it must not share the `unknown:` identity of a version
        # whose content merely could not be recovered: one says the file is gone, the other says we did
        # not see it. Collapsing them would let a deletion dedupe against a failed read of the same path.
        if deleted:
            content_cid = f"deleted:{path}"
        elif data is not None:
            content_cid = str(get_cid_for_bytes(data, False))
        else:
            content_cid = f"unknown:{path}"

        existing = self._by_content.get((path, content_cid))
        if existing is not None:
            return existing

        history = self._file_versions.setdefault(path, [])
        version = len(history) + 1
        storable, redacted = self.policy.prepare(path, data)

        asset_cls = _asset_class_for(path)
        meta = self._meta(
            **{
                prov.K_PROV_TYPE: "Entity",
                prov.K_FILE_PATH: path,
                prov.K_FILE_VERSION: version,
                prov.K_OBSERVED: event.observed,
                prov.K_USER_MODIFIED: event.user_modified,
                prov.K_REDACTED: redacted,
                prov.K_DELETED: deleted,
                "content-cid": content_cid,
            }
        )

        payload: Any
        if deleted:
            payload = {"path": path, "deleted": True}
        elif storable is None:
            # Content withheld by policy (or absent). The node still exists and still carries the true
            # content CID, so lineage is complete even though the bytes are not published. The secret
            # bytes are never handed to an asset constructor at all.
            payload = {"path": path, "content-cid": content_cid, "redacted": redacted}
        else:
            payload = storable.decode("utf-8", errors="replace")

        basename = path.rsplit("/", 1)[-1]
        asset = self._asset(asset_cls,
            payload,
            name=f"{basename} (deleted)" if deleted else f"{basename} (v{version})",
            description=(
                f"Removal of '{path}' observed during the agent session."
                if deleted
                else f"Version {version} of '{path}' observed during the agent session."
            ),
            **meta,
        )

        record = FileVersion(
            path=path,
            content_cid=content_cid,
            asset_cid=asset.cid,
            version=version,
            observed=event.observed,
            redacted=redacted,
            user_modified=event.user_modified,
        )
        history.append(record)
        self._by_content[(path, content_cid)] = record
        self._edge(record.asset_cid, prov.HAS_PATH, path, event.observed)

        previous = history[-2] if version > 1 else None
        if previous is not None:
            self._edge(record.asset_cid, prov.WAS_DERIVED_FROM, previous.asset_cid, event.observed)
            self._edge(previous.asset_cid, prov.WAS_INVALIDATED_BY, record.asset_cid, event.observed)

        run = self._runs.get(event.tool_use_id) if event.tool_use_id else None

        if run is None and event.mode != "read":
            # Nothing is open to own this transition -- a watcher event or snapshot delta for a file
            # some Bash command changed. Give it its own activity so the version is attached to the
            # session rather than left as an orphan node, and anchor a first sighting to the
            # conversation state: the change is real and unattributed, which is what observed=False
            # says. Guessing which command caused it would be a claim the data does not support.
            anchor = [previous.asset_cid] if previous is not None else (
                [self._context_anchor] if self._context_anchor is not None else []
            )
            if deleted:
                label = f"{basename}: deleted"
            elif previous is not None:
                label = f"{basename}: v{previous.version} -> v{version}"
            else:
                label = f"{basename}: appeared (v{version})"
            self._finalize(
                name=label,
                kind=prov.KIND_FILE_VERSION,
                inputs=anchor,
                outputs=[record.asset_cid],
                observed=event.observed,
                extra={prov.K_FILE_PATH: path, "attribution": "inferred" if not event.observed else "observed"},
            )
        if run is not None:
            if event.mode == "read":
                run.inputs.append(record.asset_cid)
            else:
                if version > 1:
                    run.inputs.append(history[-2].asset_cid)
                run.outputs.append(record.asset_cid)
                run.observed = run.observed and event.observed

        return record

    # ------------------------------------------------------------------ event handlers
    def _on_session_started(self, event: SessionStarted) -> None:
        self._session_id = event.session_id
        self._permission_mode = event.permission_mode
        self._effort = event.effort

        # The agent asset pins *what ran*: CLI and version, model, permission mode. Without it a manifest
        # attests a transcript rather than a computation -- there is nothing to say the run was
        # reproducible under a stated configuration.
        # Payload determines the CID, so only identity-determining facts belong in it: which agent, at
        # which version, driving which model. Session context (cwd, git branch, permission mode, effort)
        # goes to metadata instead. In the payload it would mint a different Agent asset per session --
        # and because the agent is an input to nearly every activity, two capture paths that merely
        # observed different context fields would disagree on every activity CID downstream. Two
        # sessions of the same agent build really are the same agent.
        agent = self._asset(
            Agent,
            {"agent": event.agent, "version": event.agent_version, "model": event.model},
            name=f"{event.agent} {event.agent_version or ''}".strip(),
            description="Coding agent that performed this session.",
            **self._meta(
                **{
                    prov.K_PROV_TYPE: "Agent",
                    prov.K_AGENT: event.agent,
                    prov.K_PERMISSION_MODE: event.permission_mode,
                    prov.K_EFFORT: event.effort,
                    "cwd": event.cwd,
                    "git-branch": event.git_branch,
                }
            ),
        )
        self._agent_cid = agent.cid

        anchor = self._asset(Dataset,
            # Same rule: the anchor is identified by which session and which segment of it. `source`
            # (startup/resume/clear) is context and lives in metadata.
            {"session": event.session_id, "segment": 0},
            name=f"session {event.session_id[:8]}: context",
            description="Anchor for the agent's conversation state; compaction edges attach here.",
            **self._meta(**{prov.K_PROV_TYPE: "Entity", "source": event.source}),
        )
        self._context_anchor = anchor.cid
        self._session_inputs.append(agent.cid)
        self._edge(anchor.cid, prov.WAS_ATTRIBUTED_TO, agent.cid)

    def _on_instructions_loaded(self, event: InstructionsLoaded) -> None:
        data = as_bytes(event.content) if event.content is not None else None
        storable, redacted = self.policy.prepare(event.path, data)
        payload = storable.decode("utf-8", errors="replace") if storable is not None else {"path": event.path}

        asset = self._asset(SystemPrompt,
            payload,
            name=event.path.rsplit("/", 1)[-1],
            description=f"Instructions loaded into agent context from '{event.path}'.",
            **self._meta(
                **{
                    prov.K_PROV_TYPE: "Entity",
                    prov.K_FILE_PATH: event.path,
                    prov.K_REDACTED: redacted,
                    "load-reason": event.load_reason,
                }
            ),
        )
        self._session_inputs.append(asset.cid)

    def _on_prompt_submitted(self, event: PromptSubmitted) -> None:
        asset = self._asset(Prompt,
            event.text,
            name=f"prompt {(event.prompt_id or '')[:8]}".strip(),
            description="User prompt submitted to the agent.",
            **self._meta(**{prov.K_PROV_TYPE: "Entity"}),
        )
        self._session_inputs.append(asset.cid)

    def _on_model_call(self, event: ModelCall) -> None:
        # No Prompt entity when the messages were not captured. A transcript stores the conversation,
        # not the request payload, so inventing a placeholder prompt asset would put a node in the graph
        # asserting an input that was never observed.
        prompt = None
        if event.messages_in is not None:
            prompt = self._asset(Prompt,
                to_jsonable(event.messages_in, self._hook),
                name=f"{event.model}: prompt",
                description="Messages sent to the model.",
                **self._meta(**{prov.K_PROV_TYPE: "Entity"}),
            )
        model = self._asset(Model,
            {"model": event.model},
            name=event.model,
            description="Model invoked by the agent.",
            **self._meta(**{prov.K_PROV_TYPE: "Entity"}),
        )
        output = self._asset(Reasoning,
            _payload(event.output, self._hook, "model produced no text output"),
            name=f"{event.model}: response",
            description="Model response, including any tool calls it requested.",
            **self._meta(**{prov.K_PROV_TYPE: "Entity", "stop-reason": event.stop_reason}),
        )

        inputs = [model.cid, *self._session_inputs]
        if prompt is not None:
            inputs.insert(0, prompt.cid)
        if self._context_anchor is not None:
            inputs.append(self._context_anchor)

        extra: Dict[str, Any] = {}
        if event.usage:
            extra.update(scalar_metadata({"usage": event.usage}, self._hook))
        activity = self._finalize(event.model, prov.KIND_MODEL, inputs, [output.cid], extra=extra)
        if activity is not None:
            self._last_model_activity = activity

    def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        tool_cid = self._tool_assets.get(event.tool_name)
        if tool_cid is None:
            # Registered from source when the adapter can recover it, so the Tool asset is
            # content-addressed to its implementation and a changed tool shows up as a changed asset.
            asset = self._asset(Tool,
                event.tool_source if event.tool_source is not None else {"name": event.tool_name},
                name=event.tool_name,
                description=event.tool_description or "",
                **self._meta(**{prov.K_PROV_TYPE: "Entity"}),
            )
            tool_cid = asset.cid
            self._tool_assets[event.tool_name] = tool_cid

        tool_input = self._asset(Dataset,
            _payload(event.tool_input, self._hook, "tool arguments not captured"),
            name=f"{event.tool_name}: input",
            description=f"Arguments passed to '{event.tool_name}'.",
            **self._meta(**{prov.K_PROV_TYPE: "Entity", prov.K_TOOL_USE_ID: event.tool_use_id}),
        )

        run = _Run(
            tool_use_id=event.tool_use_id,
            tool_name=event.tool_name,
            kind=prov.KIND_TOOL,
            inputs=[tool_cid, tool_input.cid],
            parent_id=event.parent_id,
        )
        if self._context_anchor is not None:
            run.inputs.append(self._context_anchor)
        self._runs[event.tool_use_id] = run

        if self._last_model_activity is not None:
            self._edge(self._last_model_activity, prov.TRIGGERED, tool_input.cid)

    def _on_tool_call_ended(self, event: ToolCallEnded) -> None:
        run = self._runs.pop(event.tool_use_id, None)
        if run is None:
            logger.debug("tool result for unknown call %s", event.tool_use_id)
            return

        result = self._asset(Dataset,
            _payload(event.result, self._hook, "tool returned no result"),
            name=f"{run.tool_name}: {'error' if event.is_error else 'output'}",
            description=f"Result returned by '{run.tool_name}'.",
            **self._meta(
                **{
                    prov.K_PROV_TYPE: "Entity",
                    prov.K_TOOL_USE_ID: event.tool_use_id,
                    "is-error": event.is_error,
                }
            ),
        )

        decision = self._pending_permissions.pop(event.tool_use_id, None)
        guardrail_cid = self._guardrail_for(decision) if decision is not None else None

        self._finalize(
            name=run.tool_name,
            kind=run.kind,
            inputs=run.inputs,
            outputs=[*run.outputs, result.cid],
            observed=run.observed,
            extra={prov.K_TOOL_USE_ID: event.tool_use_id, "is-error": event.is_error},
            authorized_by=guardrail_cid,
        )

    def _on_file_observed(self, event: FileObserved) -> None:
        self.observe_file(event)

    def _on_permission_decision(self, event: PermissionDecision) -> None:
        if event.tool_use_id is not None:
            self._pending_permissions[event.tool_use_id] = event
        else:
            self._guardrail_for(event)

    def _guardrail_for(self, event: PermissionDecision) -> Optional[CID]:
        asset = self._asset(Guardrail,
            {
                "decision": event.decision,
                "tool": event.tool_name,
                "mode": event.mode,
                "reason": event.reason,
                "source": event.source,
                "rules": event.rules,
            },
            name=f"{event.decision}: {event.tool_name or 'tool'}",
            description="Permission decision governing a tool call.",
            **self._meta(**{prov.K_PROV_TYPE: "Entity", prov.K_TOOL_USE_ID: event.tool_use_id}),
        )
        return asset.cid

    def _on_subagent_started(self, event: SubagentStarted) -> None:
        self._subagents[event.agent_id] = {
            "type": event.agent_type,
            "prompt": event.prompt,
            "model": event.model,
            "parent": event.parent_tool_use_id,
        }

    def _on_subagent_ended(self, event: SubagentEnded) -> None:
        info = self._subagents.pop(event.agent_id, {})

        spec = self._asset(Dataset,
            {"agent_id": event.agent_id, "type": info.get("type"), "prompt": info.get("prompt")},
            name=f"subagent {(info.get('type') or event.agent_id)[:24]}: input",
            description="Task delegated to a subagent.",
            **self._meta(**{prov.K_PROV_TYPE: "Entity"}),
        )
        result = self._asset(Reasoning,
            _payload(event.result, self._hook, "subagent result not captured"),
            name=f"subagent {(info.get('type') or event.agent_id)[:24]}: result",
            description="Result returned by a subagent.",
            **self._meta(**{prov.K_PROV_TYPE: "Entity", prov.K_OPAQUE: event.opaque}),
        )

        extra: Dict[str, Any] = {prov.K_OPAQUE: event.opaque}
        if event.stats:
            extra.update(scalar_metadata({"tool-stats": event.stats}, self._hook))
        if event.opaque:
            # The capture path could not see inside. Say so on the node rather than emitting an empty
            # subgraph that reads as "the subagent did nothing".
            extra["opacity-reason"] = "subagent internals not visible to this capture path"

        self._finalize(
            name=f"subagent: {info.get('type') or event.agent_id}",
            kind=prov.KIND_SUBAGENT,
            inputs=[spec.cid, *self._session_inputs],
            outputs=[result.cid],
            extra=extra,
        )

    def _on_compacted(self, event: Compacted) -> None:
        previous = self._context_anchor
        anchor = self._asset(Dataset,
            {
                "session": self._session_id,
                "post_tokens": event.post_tokens,
                "logical_parent": event.logical_parent,
            },
            name=f"session {(self._session_id or '')[:8]}: context (post-compaction)",
            description="Conversation state after a lossy context compaction.",
            **self._meta(**{prov.K_PROV_TYPE: "Entity"}),
        )
        self._context_anchor = anchor.cid

        if previous is None:
            return

        dropped = event.dropped_tokens
        if dropped is None and event.pre_tokens is not None and event.post_tokens is not None:
            dropped = event.pre_tokens - event.post_tokens

        self._finalize(
            name="context compaction",
            kind=prov.KIND_COMPACTION,
            inputs=[previous],
            outputs=[anchor.cid],
            extra={
                prov.K_EDGE_KIND: "compacted",
                "trigger": event.trigger,
                "pre-tokens": event.pre_tokens,
                "post-tokens": event.post_tokens,
                "dropped-tokens": dropped,
            },
        )
        self._edge(anchor.cid, prov.WAS_COMPACTED_FROM, previous)

    def _on_session_ended(self, event: SessionEnded) -> None:
        # Any tool call still open never reported a result. Dropping it silently would hide a failure;
        # closing it with an explicit marker keeps the count of calls honest.
        for tool_use_id in list(self._runs):
            self.handle(ToolCallEnded(tool_use_id=tool_use_id, result={"unterminated": True}, is_error=True))

    _DISPATCH = {
        "SessionStarted": _on_session_started,
        "SessionEnded": _on_session_ended,
        "InstructionsLoaded": _on_instructions_loaded,
        "PromptSubmitted": _on_prompt_submitted,
        "ModelCall": _on_model_call,
        "ToolCallStarted": _on_tool_call_started,
        "ToolCallEnded": _on_tool_call_ended,
        "FileObserved": _on_file_observed,
        "PermissionDecision": _on_permission_decision,
        "SubagentStarted": _on_subagent_started,
        "SubagentEnded": _on_subagent_ended,
        "Compacted": _on_compacted,
    }


def _payload(value: Any, hook: Optional[JsonableHook], absent: str) -> Any:
    """Coerce an event field into something ``serialize_for_hashing`` accepts.

    The SDK raises ``TypeError`` on ``None``, and ``None`` is a legitimate value throughout the event
    vocabulary -- a transcript cannot recover a model's request payload, a synthesized tool call has no
    input, an interrupted tool returns nothing. Substituting a marker keeps the node in the graph and
    says explicitly what is missing, which is the whole point of recording absence.
    """
    jsonable = to_jsonable(value, hook)
    if jsonable is None:
        return {"absent": absent}
    return jsonable


def _dedupe(cids: List[CID]) -> List[CID]:
    """Order-preserving dedupe. Duplicate inputs are common (a file read twice in one call) and would
    otherwise inflate the statement without changing its meaning."""
    seen = set()
    out = []
    for cid in cids:
        key = str(cid)
        if key not in seen:
            seen.add(key)
            out.append(cid)
    return out


__all__ = ["FileVersion", "LineageRecorder"]
