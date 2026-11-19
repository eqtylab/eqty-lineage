# Running the LangChain travel agent in OpenShell

`Dockerfile` builds an OpenShell sandbox image containing the EQTY-instrumented travel agent, its chat UI
and the research agent. `policy.yaml` limits the sandbox to OpenAI chat completions from the image's
Python interpreter.

## Build

Run the build from the repository root, not from this directory. The image installs the workspace
packages from `pyproject.toml`, `uv.lock` and `packages/`, so the build context must be the root;
`Dockerfile.dockerignore` keeps it down to the files the image needs.

```bash
docker build --platform linux/amd64 \
  -f examples/langchain/openshell/Dockerfile \
  -t eqty-langchain-agent:v2 .
```

Give each build a new tag. Kubernetes gateways pull sandbox images `IfNotPresent`, so a node that
already has a tag keeps running the image it has.

## Deploy the image

Docker and Podman gateways use the local image directly. A Kubernetes gateway needs the image on every
node that can run sandboxes, or in a registry the nodes can pull from. For k3s nodes:

```bash
docker save eqty-langchain-agent:v2 | gzip > eqty-langchain-agent-v2.tar.gz
scp eqty-langchain-agent-v2.tar.gz <node>:

# On each node:
gunzip -c eqty-langchain-agent-v2.tar.gz | sudo k3s ctr images import -
sudo k3s ctr images ls | grep eqty-langchain-agent
```

## Run

OpenShell replaces the image `CMD`, so the start command goes after `--`.

```bash
openshell sandbox create \
  --name eqty-travel-agent \
  --from eqty-langchain-agent:v2 \
  --policy examples/langchain/openshell/policy.yaml \
  --provider guardian-enrollment \
  --forward 8080 \
  -- /opt/venv/bin/python /app/travel_chat_ui.py --no-open
```

Open http://localhost:8080. For a one-shot run instead of the UI, use
`-- /opt/venv/bin/python /app/travel_agent.py "your question"` or `-- /opt/venv/bin/python /app/research_agent.py`.

Manifests and the `.eqty_sdk` store are written under `/sandbox`.

