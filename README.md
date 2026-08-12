# eqty-lineage

Workspace of EQTY lineage integrations, published to `pypi.eqtylab.io`. Each
integration lives under `packages/` as its own distribution sharing the
`eqty_lineage` import namespace:

| Package | Import | Purpose |
| --- | --- | --- |
| `eqty-lineage-langchain` | `eqty_lineage.langchain` | Callback handler registering LangChain/LangGraph runs as EQTY lineage |
| `eqty-lineage-codex` | `eqty_lineage.codex` | Replays captured Codex hook events into signed lineage |

Future integrations (e.g. `eqty-lineage-mcp`, `eqty-lineage-llamaindex`) follow
the same pattern. Because `eqty_lineage` is an implicit namespace package, no
package may ship an `eqty_lineage/__init__.py`.

## Usage

### LangChain / LangGraph

```python
from eqty_lineage.langchain import EqtyCallbackHandler

app.invoke(state, config={"callbacks": [EqtyCallbackHandler()]})
```

### Codex

A Codex plugin (`plugins/eqty-lineage`) records every hook event to a JSONL capture; the export
replays that capture into a signed manifest. See [QUICKSTART.md](QUICKSTART.md) for the full path
from an empty machine to a graph in the Explorer.

```console
$ eqty-codex-lineage /tmp/codex-hooks.jsonl
session 019f9ff2-2dd9-77f0-b687-22723134122c · gpt-5.4-mini
  1 prompt(s), 2 tool attempt(s)
    allow    executed=True  apply_patch
    deny     executed=False Bash
/tmp/codex-hooks.lineage.json
```

Output defaults to `<capture>.lineage.json`; `-o` overrides it, `--quiet` prints only the path, and
`--json` prints a machine-readable summary. `python -m eqty_lineage.codex` works if the console
script is not on your PATH.

Each tool call is recorded with the decision made about it — `allow` where it was observed to run,
`deny` where the collector refused it, and `unknown` where the capture cannot say. A denied or
unresolved call carries a guardrail node and no result node, because nothing ran. The library form
is the same thing:

```python
from eqty_lineage.codex import replay_capture

replay_capture("codex-hooks.jsonl", "session.json")
```

## Develop

```bash
just sync     # install the workspace + dev deps into .venv
just build    # build wheel + sdist for all packages into ./dist
just publish  # upload ./dist to pypi.eqtylab.io (uv publish --index eqty)
just clean    # remove ./dist
```

`eqty-sdk` resolves from the eqty index configured in `pyproject.toml`, not
public PyPI.
