# OpenShell sandbox image for the EQTY-instrumented LangChain examples.
# Build from the repository root; the workspace packages are installed from the lockfile:
#
#   docker build -f examples/langchain/openshell/Dockerfile -t eqty-langchain-agent:latest .

FROM python:3.12-slim

# iproute2 is required for OpenShell's network namespace; nftables enables bypass detection.
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 nftables \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY packages/eqty-lineage-langchain packages/eqty-lineage-langchain
COPY packages/eqty-lineage-deepagents packages/eqty-lineage-deepagents
COPY packages/eqty-lineage-vnim packages/eqty-lineage-vnim
# The default `dev` group carries langchain-openai and langgraph, which the examples import.
RUN uv sync --frozen --no-editable && rm -rf /build /root/.cache

# Code lives outside /sandbox: on Kubernetes the workspace PVC is mounted over /sandbox.
COPY examples/langchain/travel_agent.py examples/langchain/travel_chat_ui.py examples/langchain/research_agent.py /app/

RUN groupadd --gid 1500 agent \
    && useradd --uid 1500 --gid agent --home-dir /sandbox --no-create-home agent \
    && install -d -o agent -g agent /sandbox

# ChatOpenAI refuses to start without a key. Behind the EQTY supervisor, requests to api.openai.com are
# redirected to the Guardian LLM gateway; an attached provider or `--env` overrides this placeholder.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OPENAI_API_KEY=openshell-placeholder

# eqty_sdk keeps its .eqty_sdk store and signer under the working directory, so it must be writable.
WORKDIR /sandbox
USER agent

# OpenShell replaces CMD with its supervisor; pass this after `--` on `openshell sandbox create`.
CMD ["python", "/app/travel_chat_ui.py", "--no-open"]
