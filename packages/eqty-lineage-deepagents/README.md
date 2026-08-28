# eqty-lineage-deepagents

EQTY lineage callback handler for [DeepAgents](https://github.com/langchain-ai/deepagents). Records a deep agent run —
its virtual filesystem, its plan, its subagents and its skills — as EQTY data assets and computation statements.

`EqtyDeepAgentsHandler` subclasses `EqtyCallbackHandler` from
[`eqty-lineage-langchain`](../eqty-lineage-langchain/README.md), so everything that handler records is recorded here
too: graph nodes, model calls, tool calls, retrievals, subagent boundaries, failures, credential redaction. This
package adds the parts of a deep agent that live in its **state** rather than in its callbacks.

## Usage

```python
from eqty_lineage.deepagents import EqtyDeepAgentsHandler

agent.invoke({"messages": [...]}, config={"callbacks": [EqtyDeepAgentsHandler()]})
```

Use one handler instance per invocation. `deepagents` itself is never imported, and is **not** a dependency of this
package — everything is read from the callback stream and from graph state, so the handler works against whatever
version of DeepAgents produced the run, and installing it does not pin your `langchain`/`langgraph` versions.
`langchain-core` is the only framework requirement.

```bash
just deepagents-demo          # scripted model, no API key; writes manifests/deep-agent.json
```

## What it records

| DeepAgents concept | asset | where it lands in the lineage |
| --- | --- | --- |
| the deep agent | `Agent` | input to the root `graph` computation |
| a subagent | `Agent` | input to its own `agent` computation **and** to the `task` call that spawned it |
| a subagent's execution | — | an `agent` computation (from the base handler) |
| a virtual file | `Document` | output of the call that wrote it; input to every call that reads it |
| a revision of the todo list | `Dataset` | output of the `write_todos` call that wrote it, chained to the one it replaced |
| a loaded skill | `Skill` | input to the model turn whose state held the catalogue |
| a model turn's system prompt | `SystemPrompt` | input to that `chat_model` computation |

Everything above maps onto an asset type the SDK already ships; this package introduces no new primitives.

## The virtual filesystem

A deep agent carries its whole filesystem in graph state, so without an extractor every node's state asset embeds
every file — once per node — and no file is ever an entity in its own right. `VirtualFileExtractor` claims the `files`
state key and replaces it with a map of path → content CID, so the state asset says *which* files the node had
without containing them.

Versions are keyed on `(path, content)`, the same rule `PathExtractor` applies to real paths:

- **new content at a known path** — a new `Document`, recorded as an *output* of the call that wrote it, with the
  version it replaced linked as an *input*. Successive edits form a chain.
- **content already registered** — carried through as an *input*, never re-emitted as an output. A *write* that
  restores earlier bytes is the exception: it mints no new asset, but it is still that call's output, because the
  version it replaced has stopped being current.

Paths are normalized the way DeepAgents normalizes them. `validate_path` forces a leading slash and collapses `.`
and `//`, so a model that asks to write `report.md` creates `/report.md` in state; keying the raw argument would put
the write on one entity and every later read on another, and live models omit the leading slash routinely.

The path is part of the asset's payload, so the same bytes written to two paths are two files rather than one entity
wearing whichever name happened to be registered first.

### Why the write is attributed from the tool's arguments

DeepAgents' `StateBackend` applies its writes through LangGraph's channel API, not through the tool's return value.
The new content therefore appears in neither the tool's result nor the enclosing node's output state — it simply
shows up in `files` at the next model turn. Left at that, a file the agent wrote would be an *input* to everything
that later read it and an output of nothing, which in a provenance graph says the run found it already there.

So the handler reconstructs the write from the call's own arguments:

- `write_file` carries the exact content, so the new version is registered directly.
- `edit_file` reports only that it succeeded, so the result is derived from the version the edit was made against —
  a plain string replacement, with the backend's own occurrence rules re-checked before it is trusted. If any check
  fails, **nothing is registered**: the file then appears at its next sighting in state, as an input, which
  understates its provenance rather than misstating it.
- `delete` drops the path's current version, so a later write chains to nothing rather than to content that no
  longer existed, and a later read is not linked to a version it could not have read.
- a call that failed registers nothing. Failure is read from the result's `status` rather than its wording: only
  some backends say "Error", while the store, LangSmith and sandbox backends report "Failed to write file …" or the
  remote's own message verbatim, and a prefix match would take those for successes.

## The plan

`TodoListMiddleware` is **not** part of the default deep agent stack — it comes from `langchain` and has to be passed
explicitly:

```python
from langchain.agents.middleware import TodoListMiddleware

create_deep_agent(..., middleware=[TodoListMiddleware()])
```

With it, each `write_todos` call produces a `Dataset` holding that revision of the plan, chained to the revision it
replaced. A plan written once and never revised is one asset; a plan revised four times is four, in a chain. Without
the middleware the state key does not exist and nothing is recorded — no phantom plan appears.

## Skills

`SkillsMiddleware` parses each `SKILL.md` and puts the catalogue in the `skills_metadata` state key, so the skills a
model turn could draw on are exactly the ones in the state entering it. Each is registered as a `Skill` asset,
content-addressed to that metadata, and **carried** as an input — a skill is something the run was given, never
something it produced. The `SKILL.md` file itself is an ordinary virtual file, registered when it is read.

## State keys, not tool arguments

The extractors claim a state key at the top of a state and inside the `update` of a LangGraph `Command` (which is how
a tool applies a state update, and how `task` hands a subagent's work back). They deliberately do **not** claim the
same name anywhere else, because the arguments of a call and the state it produces are serialized by the same code:
`write_todos(todos=[...])` would otherwise register the new plan as an *input* to the call that wrote it, reversing
the one edge the plan's lineage exists to record.

## Writing your own extractor

`StateExtractor` is re-exported here, so a custom state key needs no separate import:

```python
from eqty_lineage.deepagents import UNCLAIMED, EqtyDeepAgentsHandler, StateExtractor


class BudgetExtractor(StateExtractor):
    def extract(self, key_path, value, sink):
        if key_path != ("budget",):
            return UNCLAIMED
        ...


handler = EqtyDeepAgentsHandler()
handler.add_extractor(BudgetExtractor())
```

Extractors are consulted newest-first, so one registered here beats this package's own. See the
[LangChain package README](../eqty-lineage-langchain/README.md) for the full contract.

## Limitation: context compaction is not in the graph

`SummarizationMiddleware` is in the default deep agent stack. On a long run it compacts the conversation: the new
context is derived from the old, but **lossily**, with most of it discarded. This handler does not record that, for
two reasons.

**It is not observable.** DeepAgents compacts inside `wrap_model_call`, not in a graph node of its own, so no
callback marks the boundary. What reaches `on_chat_model_start` is simply a shorter message list.

**There is nowhere to put it.** The SDK's edge vocabulary is `add_computation_statement(inputs, outputs)` —
`prov:used` and `prov:wasGeneratedBy` — plus `CERTIFIES`, `INCLUDES` and `IS_INSTANCE_OF` associations. Recording
compaction with any of them asserts an ordinary derivation, implying the output carries the input when in fact most
of it was dropped. Omitting it loses the boundary. Neither is honest, and no asset type fixes it: the missing thing
is an *edge*, not an entity. PROV, OpenLineage and in-toto do not model lossy derivation either.

**What this means for a reader.** On a run long enough to compact, the recorded lineage of the final answer will show
it derived from the model turns that survived compaction, and will not show that earlier context existed and was
discarded. A manifest that omitted this note would claim a completeness it does not have. An `eqty:wasCompactedFrom`
edge is the one thing here that cannot be expressed today and cannot be worked around by choosing a different asset
type; it is being raised with the SDK team separately.

Compaction only fires on long runs, so shorter runs are unaffected.

## `@eqty_tool` and the built-in tool belt

`eqty_tool` is re-exported here and works as it does in the LangChain package: decorate your own tool and
its `Tool` asset is content-addressed to its source rather than to a name/description stub.

The tools that do a deep agent's most interesting work, though, are not yours to decorate — `write_file`,
`edit_file` and `task` are built by DeepAgents' own middleware. Without their source, the manifest records
*that* a file was written but not by what code. Their source is perfectly readable, so the same function the
decorator calls can be applied to the belt the compiled agent assembled:

```python
agent = create_deep_agent(...)

for tool in agent.nodes["tools"].bound.tools_by_name.values():
    eqty_tool(tool)
```

Upgrade DeepAgents and those assets change, which is the point. `examples/deepagents/research_agent.py`
does exactly this; its manifest carries six Tool assets, each holding the source of the function that ran.
The handler does not do it for you — reaching into a compiled graph for the belt is a caller's liberty, not
something a callback handler should assume — and a tool whose source cannot be read simply falls back to
its stub.

## Verbose metadata

```python
EqtyDeepAgentsHandler(verbose=True)
```

Works exactly as in the LangChain package; see
[its README](../eqty-lineage-langchain/README.md#verbose--extra-metadata-on-assets).
