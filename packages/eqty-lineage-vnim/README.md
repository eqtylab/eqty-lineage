# EQTY VNIM Chat Models

`eqty-lineage-vnim` provides the LangChain chat client for EQTY VNIM models. It uses
the OpenAI-compatible chat-completions interface exposed by VNIM and is distributed separately so
applications can install it only when they use EQTY models:

```bash
pip install eqty-lineage-vnim
```

```python
from eqty_lineage.vnim import ChatEqtyVnimOpenAI

model = ChatEqtyVnimOpenAI(model="your-vnim-model")
```

Configure the model endpoint and credentials using the normal `ChatOpenAI` parameters or environment
variables supported by your VNIM deployment.

## Integrity manifest retrieval

`ChatEqtyVnimOpenAI` always requests response headers and reads the upstream
`X-EQTY-Request-ID` from the first streamed chunk that includes it. Once the caller consumes the
complete stream, it fetches the corresponding manifest from the server-side VNIM endpoint. Configure
that endpoint directly or with `VNIM_INTEGRITY_BASE_URL`:

```python
model = ChatEqtyVnimOpenAI(
    model="your-vnim-model",
    manifest_base_url="http://127.0.0.1:8000",
)

for chunk in model.stream(messages, config={"callbacks": callbacks}):
    ...

result = model.last_integrity_result
manifest = result.manifest if result else None
```

The manifest fetch retries only `404 Not Found` responses, every three seconds for up to 45 seconds.
The final chunk's `response_metadata` includes `eqty_request_id` and, on success,
`eqty_integrity_manifest`. It never reconstructs or alters upstream SSE bytes; the manifest is a
separate artifact.

`ChatEqtyVnimOpenAI` subclasses the pinned `langchain-openai==1.3.5` `ChatOpenAI` implementation and
adds only VNIM-specific integrity-manifest behavior. It enables `streaming=True` by default, unlike
LangChain's `ChatOpenAI`, so it can retain the upstream request ID until stream completion.

The manifest endpoint is requested without additional authentication headers. Point
`manifest_base_url` only at a trusted VNIM endpoint (the normal deployment is a local authenticated
SSH tunnel). HTTPS certificate verification remains enabled by HTTPX for HTTPS URLs.
