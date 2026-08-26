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
just demo                                 # working tree vs. the newest release tag
just demo eqty-lineage-langchain@0.0.1    # ...or any ref you name
```

Runs one identical document-review agent twice — once through the handler at the baseline ref, once through
the working tree — and writes `manifests/before.json` and `manifests/after.json`. The model is scripted rather
than live, so the two runs are byte-identical and every difference in the manifest is attributable to the
handler alone.

The baseline defaults to the newest `eqty-lineage-langchain@*` tag, so this keeps answering "what changed
since the last release" as releases are cut, rather than freezing into a comparison against one fixed version.
Rows that differ are marked `*`; a run against an unchanged baseline marks nothing.

Every row is a property of the lineage and is stable run to run. The manifest's raw statement and asset
counts are deliberately not reported: assets are content-addressed, so two state payloads that happen to
coincide collapse into one registration and the totals move by one between otherwise identical runs.

Against `0.0.1`, the exported manifests look like this in the graph explorer.

(The screenshots below predate the retrieval and subagent steps, so they show the earlier, smaller graph.)

Before — only `Dataset` and `Tool` assets, one red tool wrench, and `verify_b`'s output going nowhere:

![lineage before](docs/images/lineage-before.png)

After — `Prompt`, `Model` and `Reasoning` appear on the left (the model call that was being dropped), a
second tool wrench for the tool that fails, and both parallel branches feeding the next node:

![lineage after](docs/images/lineage-after.png)

Or read the printed table:

| | before | after |
| --- | --- | --- |
| exceptions swallowed by LangChain | `KeyError('state_in')`, `TypeError(... NoneType)` | none |
| computations recorded | 11 | 16 |
| retrievals recorded | 0 | 1 |
| documents registered | 0 | 2 |
| subagents recorded | none | `specialist` |
| graph nodes present | `checkpoint` missing | all nine |
| computation kinds | `graph`, `graph_node`, `tool` | plus `agent`, `chat_model`, `retriever`, `tool_error` |
| orphaned node outputs | `verify_b`, `assess` | none |
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
