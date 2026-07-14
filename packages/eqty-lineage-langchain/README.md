# eqty-lineage-langchain

EQTY lineage callback handler for LangChain and LangGraph. Registers graph nodes,
chat model calls, and tool calls as EQTY data assets and computation statements,
threaded together into one end-to-end lineage flow.

## Install

```bash
pip install eqty-lineage-langchain --extra-index-url https://pypi.eqtylab.io/simple/
```

## Usage

```python
from eqty_lineage.langchain import EqtyCallbackHandler

app.invoke(state, config={"callbacks": [EqtyCallbackHandler()]})
```

Only `langchain-core` is required at runtime, so the handler works with plain
LangChain runnables as well as LangGraph graphs. Use one handler instance per
invocation.

`pathlib.Path` values in graph state get special treatment: if the path exists,
the file or directory is registered as its own Dataset asset via
`Dataset.from_path` (CIDing full directory contents) and linked into the
computation that carried it. Keep filesystem references in state as `Path`
objects rather than strings to opt in.
