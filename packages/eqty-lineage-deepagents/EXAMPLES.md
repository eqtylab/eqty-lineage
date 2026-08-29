# Instrumenting the upstream DeepAgents examples

[langchain-ai/deepagents](https://github.com/langchain-ai/deepagents/tree/main/examples) ships fifteen examples.
This page says which of them `eqty-lineage-deepagents` can record, what to change in each, and what the resulting
manifest does and does not contain.

Surveyed against upstream commit `a1af029` (2026-08-28), `deepagents` 0.7.11.

**Eight can be instrumented; seven cannot.** The seven are not a shortcoming of this package — they have no
in-process Python entrypoint for a callback to attach to. Nothing in the eight *breaks*: no example errors, and
none produces a wrong manifest. Where recording is incomplete the failure is silent under-recording, never
misrecording, which is the direction the handler is built to fail in.

## The survey

| Example | Backend | Where you attach | Recorded |
|---|---|---|---|
| `deep_research` | default `StateBackend` | `agent.py:54` — module-level graph | Everything |
| `rubric_middleware` | default `StateBackend` | `rubric_agent.py:162` — `agent.invoke` | Everything |
| `async-subagent-server` | default `StateBackend` | `server.py:174`, `supervisor.py:95` — `ainvoke` | Everything |
| `content-builder-agent` | `FilesystemBackend` | `content_writer.py:258` — `agent.astream` | Files via tool calls |
| `text-to-sql-agent` | `FilesystemBackend` | `agent.py:86` — `agent.invoke` | Files via tool calls |
| `better-harness` | `FilesystemBackend(virtual_mode=True)` | `better_harness/agent.py:214`, `:616` | Files via tool calls |
| `llm-wiki` | `CompositeBackend` — LangSmith sandbox + filesystem | `helpers.py:639` — `agent.invoke` | Files via tool calls |
| `nvidia_deep_agent` | `CompositeBackend` — Modal sandbox | `src/agent.py:90` — module-level graph | Files via tool calls |
| `ralph_mode` | — | **nothing to attach to** — runs in a server subprocess | — |
| `deploy-coding-agent` | — | **nothing to attach to** — `agent.json`, platform-hosted | — |
| `deploy-content-writer` | — | **nothing to attach to** — `agent.json`, platform-hosted | — |
| `deploy-gtm-agent` | — | **nothing to attach to** — `agent.json`, platform-hosted | — |
| `deploy-mcp-docs-agent` | — | **nothing to attach to** — `agent.json`, platform-hosted | — |
| `talon` | — | **nothing to attach to** — Docker runtime | — |
| `downloading_agents` | — | **nothing to attach to** — ships a `.zip` | — |

## The pattern

Every one of the eight is the same edit. Initialise the SDK once at startup, build one handler per conversation,
and pass it in `callbacks`:

```python
from eqty_sdk import Context, Signer, init, set_active_signer

from eqty_lineage.deepagents import EqtyDeepAgentsHandler

init(default_context=Context.new("Deep Research")).set_store_all_blobs(True)
set_active_signer(Signer.new(name="deep_research", _load_if_exists=True))

handler = EqtyDeepAgentsHandler()
result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"callbacks": [handler]},
)
```

No node, subagent or middleware is decorated, and `deepagents` is never imported by this package. `init()` is
process-global, and a second call is a silent no-op: it logs `Config already initialized` and leaves the first
store in place. So it belongs at startup rather than next to the invoke — a per-run `init()` does not give each
run its own store, it is simply ignored.

`ainvoke` and `astream` need no different treatment. The handler implements only the sync callbacks, and
LangChain dispatches those from async runs — three of the eight examples are async and record identically.

## Group A — the default backend records everything

`deep_research`, `rubric_middleware`, `async-subagent-server`.

These leave `backend` unset, so the deep agent's filesystem lives in graph state under `files`. The extractors see
it directly, and you get the full picture: every file version chained write-to-edit-to-read, the subagents `task`
delegates to, the skills catalogue, and each tool call as its own computation.

### `deep_research`

The example has no invoke of its own — `agent.py:54` builds a module-level graph that `langgraph.json` exposes as
`research` for `langgraph dev`. Callbacks are in-process Python objects and cannot be handed to a graph running
under the dev server, so drive it yourself:

```python
from agent import agent  # examples/deep_research/agent.py

handler = EqtyDeepAgentsHandler()
agent.invoke(
    {"messages": [{"role": "user", "content": "Compare the two approaches."}]},
    config={"callbacks": [handler]},
)
```

Its `tavily_search` and `think_tool` are ordinary `@tool` functions in `research_agent/tools.py`, so decorate them
with `@eqty_tool` to content-address them to their source — see [Tool sources](#tool-sources).

**Expect no plan.** `research_agent/prompts.py:7` instructs the model to "Create a todo list with `write_todos`",
but the example never installs `TodoListMiddleware`, so no `write_todos` tool exists and no plan asset appears.
That middleware is not part of the default deep agent stack; it comes from `langchain` and must be passed
explicitly. Add it and the plan is recorded with no other change.

### `rubric_middleware`

A single `agent.invoke` at `rubric_agent.py:162` — pass `config={"callbacks": [EqtyDeepAgentsHandler()]}` there.

It uses `checkpointer=InMemorySaver()` (`:157`). See [Checkpointed threads](#checkpointed-threads).

### `async-subagent-server`

Two entrypoints, and the one place in the whole set where handler lifetime is a correctness question rather than a
detail. `server.py` fans out **concurrent runs over one shared agent** (`:174`), so the handler must be built
inside the request, not once at module scope:

```python
async def execute_run(user_message: str) -> None:
    with graph_context(ctx):  # eqty_sdk.context
        await _agent.ainvoke(
            {"messages": [HumanMessage(user_message)]},
            config={"callbacks": [EqtyDeepAgentsHandler()]},  # one per run
        )
```

Sharing one handler across concurrent runs corrupts the manifest quietly: file and plan versions are keyed per
run, so a second run's write of `/report.md` is recorded as a revision of the first run's file, between runs that
share nothing. The graph still looks well-formed. It is simply wrong. The handler warns when it sees two runs open
at once, but building it per run is the fix. The agent itself holds no lineage state and can stay shared, and
`graph_context` is a `ContextVar`, so entering it inside each task is safe under `asyncio`.

`supervisor.py:95` is a single run, but uses a `MemorySaver` checkpointer — see below.

## Group B — a filesystem backend still records writes

`content-builder-agent`, `text-to-sql-agent`, `better-harness`.

These pass a `FilesystemBackend`, so files live on disk and **never appear in graph state** — state keys are just
`['messages']`, and the virtual-file extractor sees nothing. Recording still works, because write attribution is
reconstructed from the **tool's arguments**, which are backend-agnostic:

```
write_file   out=[state, /report.md v1]
edit_file    in=[..., /report.md v1]  out=[state, /report.md v2]     ← chain intact
read_file    in=[..., /report.md v2]                                 ← links the version read
```

A file the agent only ever *reads* — every pre-existing source file under this backend — is recorded too, as the
**rendering** `read_file` returned rather than as the file itself. That distinction is deliberate and is explained
in [the README](README.md#files-the-run-only-reads); the short version is that the tool result is line-numbered
and truncated, so recording it under the file's own identity would assert a content hash the file never had.

Two things to know:

- The manifest records the **virtual** path the agent used (`/report.md`), not the real one
  (`<root_dir>/report.md`). That is what the agent saw, but a reader may expect otherwise.
- `content-builder-agent` passes `skills=["./skills/"]` and `text-to-sql-agent` passes `skills=[...]`. Skills are
  captured either way, because `skills_metadata` is a state key regardless of backend.

`better-harness` reaches `create_deep_agent` through a dynamically imported module (`agent.py:203`, `:607`) and
invokes at `:214` and `:616`; both call sites take the same `config` argument. Note `:214` retries up to three
times — each attempt is a separate run, so build the handler inside the retry loop, not outside it.

## Group C — sandboxes record the same way, plus `execute`

`llm-wiki`, `nvidia_deep_agent`.

Both use a `CompositeBackend` routing some paths to a sandbox. File recording behaves exactly as in Group B — the
tool arguments are the same whatever executes them. The addition is `execute`, which is recorded as an ordinary
tool computation: command in, output out. That is the default path rather than anything sandbox-aware.

**Know its limit.** `execute` runs a shell, and a shell can write files. The handler reconstructs file versions
only from the file tools (`write_file`, `edit_file`, `delete`, `read_file`), so a file that `execute` creates or
rewrites gets no version of its own. Under the default `StateBackend` such a change still surfaces through the
`files` state key, but under exactly these sandbox and filesystem backends — which keep the filesystem out of
state — it is recorded nowhere, and a later `edit_file` on that path is reconstructed against content that never
saw the shell's change. Prefer the file tools over shell redirection when you want the writes attested.

`llm-wiki` (`helpers.py:622`) routes `/raw/`, `/wiki/`, `/log.md` and `/AGENTS.md` to a local
`FilesystemBackend(virtual_mode=True)` and everything else to a LangSmith sandbox; attach at the `agent.invoke` on
`:639`.

`nvidia_deep_agent` is a module-level graph like `deep_research`, so import and drive it yourself.
**One caution:** `src/agent.py:85` and `:98` carry a commented-out `interrupt_on={"execute": True}`. Human-in-the-loop
interrupts are untested here, and a `GraphInterrupt` is an exception — the base handler would most likely record a
pause as an error computation rather than as a pause. Check that before enabling it.

## Things that catch people out

### Checkpointed threads

`rubric_middleware` and `supervisor.py` use checkpointers. Resuming a thread is a **new invocation**, and reusing
one handler across the turns of a single thread is the deliberate, useful case: it is what chains a file written in
turn one to an edit in turn three rather than leaving two unrelated entities.

If you build a fresh handler per turn instead, files written in an earlier turn are registered on first sighting as
inputs rather than chained to their real origin. Neither is wrong; one handler per conversation is the documented
contract, and this is what that contract buys.

### Tool sources

A callback only ever carries a tool's *name*, so without help every Tool asset is a name-and-description stub. For
tools the example defines — `tavily_search`, `think_tool`, the SQL tools — add `@eqty_tool` at the definition:

```python
from eqty_lineage.deepagents import eqty_tool


@eqty_tool
@tool
def think_tool(reflection: str) -> str: ...
```

The built-in belt (`write_file`, `edit_file`, `task`) is built by DeepAgents' middleware and is not yours to
decorate. `examples/deepagents/research_agent.py` in this repo shows the same function applied to the compiled
agent's belt, which content-addresses those to their real implementations.

### Concurrency

Covered under `async-subagent-server` above, and the single most likely way to get a wrong manifest out of this
package. One handler per run.

## The seven that cannot be instrumented

`ralph_mode` delegates to `deepagents-cli`'s `run_non_interactive`, which starts a **`langgraph dev` server
subprocess** and drives it over the SDK. It builds its own `RunnableConfig` internally and exposes no callbacks
parameter, and even if it did, a callback is an in-process Python object that cannot cross into another process.

`deploy-coding-agent`, `deploy-content-writer`, `deploy-gtm-agent` and `deploy-mcp-docs-agent` are `agent.json`
manifests deployed with `deepagents deploy` and run on LangSmith's platform. `talon` ships a Dockerfile and
compose runtime; `downloading_agents` ships a `.zip`. In none of them is there a Python call site to pass
`config={"callbacks": [...]}` to, and callbacks are not serialisable across a platform API.

This is a deployment-model limitation rather than anything about the handler. Recording a platform-hosted run
would need a hook the platform exposes; we do not claim support for one today.

## What was verified, and what was not

The behaviour above was established by running this package's handler against the same backend classes the
examples use — `StateBackend` and `FilesystemBackend` are both exercised by the test suite in `tests/`, including
the write/edit/read chain, read-only renderings and the concurrency warning.

The examples themselves were **read, not executed**: nearly all require paid API keys (Anthropic, Tavily,
LangSmith, NVIDIA, Modal). The sandbox behaviour of `llm-wiki` and `nvidia_deep_agent` is therefore reasoned from
the backend contract rather than observed, and `interrupt_on` is untested in any configuration.
