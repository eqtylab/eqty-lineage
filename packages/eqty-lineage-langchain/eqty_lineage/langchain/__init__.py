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
"""

import functools
import inspect
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from eqty_sdk import CID, Dataset, Model, Prompt, Reasoning, Tool, get_cid_for_path
from eqty_sdk.metadata import Metadata
from eqty_sdk.statements import add_computation_statement

logger = logging.getLogger("eqty.langgraph")


def _to_jsonable(obj: Any, on_path: Optional[Callable[[Path], None]] = None) -> Any:
    """Convert LangChain/LangGraph values into plain JSON-serializable data.

    ``on_path`` is invoked for every existing ``pathlib.Path`` encountered, so the caller can register the
    file/directory as its own EQTY asset.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        if on_path is not None and obj.exists():
            on_path(obj)
        return str(obj)
    if isinstance(obj, BaseMessage):
        data: Dict[str, Any] = {"role": obj.type, "content": _to_jsonable(obj.content, on_path)}
        tool_calls = getattr(obj, "tool_calls", None)
        if tool_calls:
            data["tool_calls"] = _to_jsonable(tool_calls, on_path)
        usage = getattr(obj, "usage_metadata", None)
        if usage:
            data["usage"] = _to_jsonable(usage, on_path)
        return data
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, on_path) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(item, on_path) for item in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump(), on_path)
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
        # tool name -> Tool asset CID, so each tool is registered once
        self._tool_cids: Dict[str, CID] = {}
        # (resolved path, content CID) -> Dataset CID. Keyed on the *contents* as well as the path: the same
        # bytes at the same path are one entity, but rewritten bytes are a new version, and keying on the path
        # alone would attest the previous content for every computation that ran after the change.
        self._path_versions: Dict[Tuple[str, str], CID] = {}
        # resolved path -> Dataset CID of its most recent version, so a rewrite can be linked to what it replaced
        self._path_latest: Dict[str, CID] = {}

    ##################################################   Helpers   #################################################
    # kwargs the SDK asset constructors claim for themselves; verbose metadata must not shadow them
    _RESERVED_SDK_KWARGS = frozenset({"obj", "path", "name", "description", "_store"})

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
        for key, value in fields.items():
            safe_key = key
            while safe_key in self._RESERVED_SDK_KWARGS or safe_key in out:
                safe_key = f"LC-{safe_key}"
            jsonable = _to_jsonable(value)
            out[safe_key] = jsonable if isinstance(jsonable, (str, int, float, bool)) else json.dumps(jsonable)
        return out

    def _register_state(
        self, obj: Any, name: str, description: str, extra: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dataset, List[CID], List[CID]]:
        """Register ``obj`` as a Dataset; existing Paths inside it become their own assets.

        Returns ``(state_asset, carried_path_cids, created_path_cids)``. A path whose contents this run has not
        seen before is a new version *created* by the current computation; a path whose exact bytes are already
        registered is merely *carried* through the state and must be linked as an input, never re-emitted as an
        output (which would create a cycle in the lineage graph).

        Versions are keyed on ``(path, content CID)``, not on the path alone. A file that is rewritten between two
        nodes is a genuinely different entity, and reusing the first sighting's asset would attest content that the
        later computation never saw. The cost is re-hashing each path per sighting, which is what
        ``Dataset.from_path`` would do anyway on a miss.
        """
        carried: List[CID] = []
        created: List[CID] = []
        extra = self._verbose_metadata(extra or {})

        def collect(path: Path) -> None:
            key = str(path.resolve())
            try:
                content_cid = str(get_cid_for_path(path))
            except Exception:  # noqa: BLE001 - an unreadable path must not take down the run being observed
                logger.debug("could not compute a content CID for '%s'; skipping", path)
                return

            known = self._path_versions.get((key, content_cid))
            if known is not None:
                if known not in carried and known not in created:
                    carried.append(known)
                return

            asset = Dataset.from_path(
                path,
                name=path.name,
                description=f"Filesystem asset referenced by LangGraph state: '{path}'.",
                **extra,
            )
            self._path_versions[(key, content_cid)] = asset.cid

            # the version this one replaced is an input to whatever produced it, which is what makes the
            # successive versions of a file a chain in the graph rather than unrelated assets
            previous = self._path_latest.get(key)
            if previous is not None and previous not in carried and previous not in created:
                carried.append(previous)
            self._path_latest[key] = asset.cid

            if asset.cid not in created:
                created.append(asset.cid)

        payload = _to_jsonable(obj, on_path=collect)
        asset = Dataset.from_object(payload, name=name, description=description, **extra)

        return asset, carried, created

    def _finalize(self, name: str, kind: str, input_cids: List[CID], output_cids: List[CID]) -> None:
        """Create the computation node w/ metadata."""

        statement_ids = add_computation_statement(inputs=input_cids, outputs=output_cids)

        Metadata(name=name, computation_type=kind, framework="langgraph").create_statement(statement_ids[0], None, None)

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
        self._parents[run_id] = parent_run_id

        # LangGraph emits many internal chain runs (channel reads/writes, task wrappers).
        # A node-level run is the one whose run name equals the "langgraph_node" metadata entry;
        # the root run is the graph itself.
        node = (metadata or {}).get("langgraph_node")
        name = kwargs.get("name") or (serialized or {}).get("name")
        is_node = node is not None and name == node
        is_graph = parent_run_id is None

        if not (is_node or is_graph):
            logger.info("is_node and is_graph are None")
            return

        label = node if is_node else (name or "graph")

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

        self._runs[run_id] = {
            "name": label,
            "kind": "graph" if is_graph else "graph_node",
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

        if run["kind"] == "graph":
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

    @_synchronized
    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        logger.warning(run_id)
        self._parents.pop(run_id, None)
        self._runs.pop(run_id, None)
        self._forget_run(run_id)

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
        params = kwargs.get("invocation_params") or {}
        model_name = params.get("model") or params.get("model_name") or "unknown-model"

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

        model = Model.from_object(
            {"model": model_name, "provider": params.get("_type", "unknown")},
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
            # the prompt is derived from the node's input state
            input_cids.append(node["state_in"])

        self._runs[run_id] = {
            "name": model_name,
            "kind": "chat_model",
            "inputs": input_cids,
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
            run["node"]["child_outputs"].append(output.cid)

    @_synchronized
    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        logger.warning(run_id)
        self._runs.pop(run_id, None)

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

        if tool_name not in self._tool_cids:
            # an @eqty_tool-decorated tool is registered from its source code, like @compute's code asset;
            # an undecorated tool falls back to a name/description stub
            source = _registered_tool_sources.get(tool_name)
            tool_asset = Tool.from_object(
                source if source is not None else {"name": tool_name},
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
            self._tool_cids[tool_name] = tool_asset.cid

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

        input_cids = [self._tool_cids[tool_name], tool_input.cid, *carried, *created]

        node = self._enclosing_node(parent_run_id)
        if node is not None:
            # the tool call was requested by the state entering the node
            input_cids.append(node["state_in"])

        self._runs[run_id] = {
            "name": tool_name,
            "kind": "tool",
            "inputs": input_cids,
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

        input_cids = run["inputs"] + [c for c in carried if c not in run["inputs"]]
        output_cids = [tool_output.cid, *created]
        self._finalize(run["name"], run["kind"], input_cids, output_cids)

        if run["node"] is not None:
            run["node"]["child_outputs"].extend(output_cids)

    @_synchronized
    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        logger.warning(run_id)
        self._runs.pop(run_id, None)


################################################## Tool Calls ##################################################

__all__ = ["EqtyCallbackHandler", "eqty_tool"]
