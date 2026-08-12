# eqty-lineage-langchain

EQTY lineage callback handler for LangChain and LangGraph. Registers graph nodes, chat model calls, and tool calls as
EQTY data assets and computation statements, threaded together into one end-to-end lineage flow:

- every graph node run → input/output Dataset assets + a computation statement
- every chat model call → Prompt + Model assets in, Reasoning asset out + computation
- every tool call → Tool + input Dataset in, output Dataset out + computation

## Usage

```python
from eqty_lineage.langchain import EqtyCallbackHandler

app.invoke(state, config={"callbacks": [EqtyCallbackHandler()]})
```

Only `langchain-core` is required at runtime, so the handler works with plain LangChain runnables as well as LangGraph
graphs. Use one handler instance per invocation.

## `verbose` — extra metadata on assets

```python
EqtyCallbackHandler(verbose=True)
```

By default each asset carries only its name and description. With `verbose=True`, every registered asset also gets the
raw LangChain callback context as metadata: which callback produced it (`callback`), the `run_id` and `parent_run_id`,
`tags`, the LangGraph `metadata` dict, and any other keyword arguments the callback received.

Values are sanitized before they reach the SDK so they render correctly in the graph explorer (which displays each
metadata value as a string):

- non-JSON types (`UUID`, `Path`, messages, ...) are converted to strings automatically — no need to pre-stringify `run_id`
- dicts and lists are JSON-encoded (otherwise they'd display as  `[object Object]`)
- `None` is kept and encoded as the string `"null"`, so a present-but-empty key is distinguishable from an absent one
- keys that collide with the SDK's own constructor kwargs (`name`, `description`, ...) are prefixed with `LC-`
(e.g. the LangChain run name appears as `LC-name`)

## `@eqty_tool` — capture tool source code

```python
from langchain_core.tools import tool
from eqty_lineage.langchain import eqty_tool

@tool
@eqty_tool
def search(query: str) -> str:
    """Search the knowledge base."""
    ...
```

Decorating a tool function with `@eqty_tool` records its Python source code at definition time (callbacks only receive
the tool's *name* at runtime, so the source cannot be recovered later). When the tool is first called, the handler
registers its Tool asset **from the source code itself** — mirroring how `@eqty_sdk.compute` registers a code asset for
the function it wraps — so the asset's CID is content-addressed to the implementation: edit the tool's code and the
next run produces a new Tool asset, making it visible in the lineage which version of the code each computation used.

Notes:

- the decorator returns the function unchanged and works on either side of LangChain's `@tool`
- tools without the decorator still get a Tool asset, registered from a name/description stub instead of source
- source capture uses `inspect.getsource`, so it only works for functions defined in real files
(not a REPL or `exec`'d code); the tool's registered source includes its decorator lines

## `Path` values in graph state

`pathlib.Path` values in graph state get special treatment: if the path exists, the file or directory is registered as
its own Dataset asset via `Dataset.from_path` (CIDing full directory contents) and linked into the computation that
carried it. A path first seen in a computation's output is recorded as *created* by it; a path seen before is linked as
an input. Keep filesystem references in state as `Path` objects rather than strings to opt in.
