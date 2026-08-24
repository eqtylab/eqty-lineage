# eqty-lineage

Workspace of EQTY lineage integrations, published to `pypi.eqtylab.io`. Each
integration lives under `packages/` as its own distribution sharing the
`eqty_lineage` import namespace:

| Package | Import | Purpose |
| --- | --- | --- |
| `eqty-lineage-langchain` | `eqty_lineage.langchain` | Callback handler registering LangChain/LangGraph runs as EQTY lineage |

Future integrations (e.g. `eqty-lineage-mcp`, `eqty-lineage-llamaindex`) follow
the same pattern. Because `eqty_lineage` is an implicit namespace package, no
package may ship an `eqty_lineage/__init__.py`.

## Usage

```python
from eqty_lineage.langchain import EqtyCallbackHandler

app.invoke(state, config={"callbacks": [EqtyCallbackHandler()]})
```

## Demo: what the lineage looks like before and after

```bash
just demo
```

Runs one identical document-review agent twice — once through the handler released as
`eqty-lineage-langchain@0.0.1`, once through the working tree — and writes `manifests/before.json` and
`manifests/after.json`. The model is scripted rather than live, so the two runs are byte-identical and every
difference in the manifest is attributable to the handler alone.

Load both manifests in the graph explorer to compare the topology, or read the printed table:

| | before (0.0.1) | after |
| --- | --- | --- |
| exceptions swallowed by LangChain | `KeyError('state_in')`, `TypeError(... NoneType)` | none |
| computations recorded | 7 | 10 |
| graph nodes present | `checkpoint` missing | all six |
| computation kinds | no `chat_model`, no `tool_error` | both present |
| orphaned node outputs | `verify_b` | none |
| `report.md` versions tracked | 1 | 2 |
| `publish` linked to the bytes it read | **no** | yes |

## Develop

```bash
just sync     # install the workspace + dev deps into .venv
just build    # build wheel + sdist for all packages into ./dist
just publish  # upload ./dist to pypi.eqtylab.io (uv publish --index eqty)
just clean    # remove ./dist
```

`eqty-sdk` resolves from the eqty index configured in `pyproject.toml`, not
public PyPI.
