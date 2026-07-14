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
