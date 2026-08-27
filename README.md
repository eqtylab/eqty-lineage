# eqty-lineage

Workspace of EQTY lineage integrations, published to `pypi.eqtylab.io`. Each
integration lives under `packages/` as its own distribution sharing the
`eqty_lineage` import namespace:

| Package | Import | Purpose |
| --- | --- | --- |
| `eqty-lineage-langchain` | `eqty_lineage.langchain` | Callback handler registering LangChain/LangGraph runs as EQTY lineage |
| `eqty-lineage-deepagents` | `eqty_lineage.deepagents` | The same, extended with a deep agent's filesystem, plan, subagents and skills |

Future integrations (e.g. `eqty-lineage-mcp`, `eqty-lineage-llamaindex`) follow
the same pattern. Because `eqty_lineage` is an implicit namespace package, no
package may ship an `eqty_lineage/__init__.py`.

## Usage

```python
from eqty_lineage.langchain import EqtyCallbackHandler

app.invoke(state, config={"callbacks": [EqtyCallbackHandler()]})
```

Each package documents its own usage and demos. For LangChain and LangGraph, see
[`packages/eqty-lineage-langchain`](packages/eqty-lineage-langchain/README.md), which includes a
before/after lineage comparison you can run with `just langchain-diff-demo`. For DeepAgents, see
[`packages/eqty-lineage-deepagents`](packages/eqty-lineage-deepagents/README.md) and `just deepagents-demo`.

## Develop

```bash
just sync     # install the workspace + dev deps into .venv
just build    # build wheel + sdist for all packages into ./dist
just publish  # upload ./dist to pypi.eqtylab.io (uv publish --index eqty)
just clean    # remove ./dist
```

`eqty-sdk` resolves from the eqty index configured in `pyproject.toml`, not
public PyPI.
