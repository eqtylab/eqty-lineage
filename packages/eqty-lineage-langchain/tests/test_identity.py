"""What identifies a model or a tool, as opposed to what it produced.

Both used to be recorded by name alone, so two models differing only in sampling — a materially different
computation — shared one asset, and a tool configured two ways shared one too. Everything needed was
already reaching the callbacks and being discarded.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph


class S(TypedDict):
    trail: Annotated[list, lambda a, b: a + b]


def _graph(fn):
    graph = StateGraph(S)
    graph.add_node("work", fn)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


# ------------------------------------------------------------- models ----
def test_sampling_parameters_are_part_of_the_model(recording_handler):
    params = recording_handler._sampling_params(
        {"model": "gpt-4o-mini", "model_name": "gpt-4o-mini", "temperature": 0.7, "max_completion_tokens": 512}
    )
    assert params == {"temperature": 0.7, "max_completion_tokens": 512}


def test_bound_tools_are_not_part_of_the_model(recording_handler):
    """The tool belt is the agent's shape, and each tool is already its own asset."""
    params = recording_handler._sampling_params(
        {"model": "x", "temperature": 0.0, "tools": [{"function": {"name": "add", "parameters": {}}}]}
    )
    assert "tools" not in params
    assert params == {"temperature": 0.0}


def test_credentials_never_reach_the_payload(recording_handler):
    """Asset payloads are stored as blobs, so a leak here is a credential written to disk."""
    params = recording_handler._sampling_params(
        {
            "temperature": 0.0,
            "api_key": "sk-SECRET",
            "openai_api_key": "sk-SECRET",
            "auth_token": "SECRET",
            "some_secret": "SECRET",
        }
    )
    assert params["temperature"] == 0.0
    assert "SECRET" not in str(params), params


def test_top_k_is_not_mistaken_for_a_credential(recording_handler):
    """The secret hints match substrings; make sure an ordinary sampling knob survives."""
    assert recording_handler._sampling_params({"top_k": 40, "top_p": 0.9}) == {"top_k": 40, "top_p": 0.9}


# -------------------------------------------------------------- tools ----
def test_one_tool_configured_two_ways_is_two_assets(recording_handler):
    @tool
    def fetch(url: str) -> str:
        """Fetch a URL."""
        return url

    def worker(state: S) -> dict:
        staging = fetch.with_config(metadata={"endpoint": "https://staging"})
        prod = fetch.with_config(metadata={"endpoint": "https://prod"})
        return {"trail": [staging.invoke({"url": "a"}), prod.invoke({"url": "b"})]}

    _graph(worker).invoke({"trail": []}, config={"callbacks": [recording_handler]})

    assert len(recording_handler._tool_cids) == 2, "the two configurations collapsed into one asset"


def test_an_unconfigured_tool_keeps_the_bare_payload(recording_handler):
    """No caller config means the payload shape is unchanged from before."""
    import eqty_lineage.langchain as mod

    @tool
    def plain(x: str) -> str:
        """Plain."""
        return x

    def worker(state: S) -> dict:
        return {"trail": [plain.invoke({"x": "1"})]}

    _graph(worker).invoke({"trail": []}, config={"callbacks": [recording_handler]})

    key = next(iter(recording_handler._tool_cids))
    assert '"name": "plain"' in key or "def plain" in key, key
    assert mod is not None


def test_caller_metadata_reaches_the_model_asset(recording_handler, monkeypatch):
    import eqty_lineage.langchain as mod

    recorded: list = []
    real = mod.Model._from_object

    def spy(obj, asset_type, ctx=None, _store=None, **kwargs):
        recorded.append(obj)
        return real(obj, asset_type, ctx, _store, **kwargs)

    # `_asset_factory` binds a context and calls `_from_object` directly, bypassing the public
    # `from_object` classmethod entirely, so that's the point that must be patched.
    monkeypatch.setattr(mod.Model, "_from_object", classmethod(lambda cls, *a, **kw: spy(*a, **kw)))

    model = GenericFakeChatModel(messages=iter([AIMessage("hi")]))
    model.with_config(tags=["prod"], metadata={"deployment": "eu-west"}).invoke(
        "q", config={"callbacks": [recording_handler]}
    )

    assert recorded, "no Model asset registered"
    payload: Any = recorded[0]
    assert payload["config"] == {"deployment": "eu-west"}
    assert payload["tags"] == ["prod"]


# --------------------------------------------------- framework noise ----
def test_framework_tags_are_not_caller_intent(recording_handler):
    """seq:step:N and graph:step:N encode a position, so they would mint an asset per position."""
    kept = recording_handler._caller_tags(
        ["seq:step:1", "graph:step:4", "map:key:docs", "langsmith:hidden", "corpus:legal", "prod"]
    )
    assert kept == ["corpus:legal", "prod"]


def test_the_same_tool_at_two_positions_is_one_asset(recording_handler):
    """A tool called twice in one node picks up different seq:step tags; it is still one tool."""

    @tool
    def echo(x: str) -> str:
        """Echo."""
        return x

    def worker(state: S) -> dict:
        return {"trail": [echo.invoke({"x": "a"}), echo.invoke({"x": "b"})]}

    _graph(worker).invoke({"trail": []}, config={"callbacks": [recording_handler]})
    assert len(recording_handler._tool_cids) == 1, recording_handler._tool_cids


def test_max_tokens_is_a_sampling_knob_not_a_credential(recording_handler):
    """`max_completion_tokens` contains "token"; matching substrings would silently drop it."""
    for name in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        assert recording_handler._sampling_params({name: 512}) == {name: 512}, name
    redacted = recording_handler._REDACTED
    for name in ("api_key", "openai_api_key", "auth_token", "bearer_token", "some_secret", "authorization"):
        assert recording_handler._sampling_params({name: "SECRET"}) == {name: redacted}, name
    # ...and words that merely look similar are not credentials
    for name in ("author", "authoring_model", "top_k", "keyword_boost"):
        assert recording_handler._sampling_params({name: "kept"}) == {name: "kept"}, name


# ------------------------------------------------------ redaction ----
def test_caller_metadata_is_redacted(recording_handler):
    """`with_config(metadata={"api_key": ...})` reaches an asset payload directly."""
    kept = recording_handler._caller_metadata({"api_key": "sk-SECRET", "index": "legal"})
    assert kept["index"] == "legal", "ordinary config must survive"
    assert "SECRET" not in str(kept), kept


def test_nested_credentials_are_redacted(recording_handler):
    """A credential can be nested arbitrarily; checking only top-level keys misses it."""
    params = recording_handler._sampling_params({"temperature": 0.0, "extra_body": {"api_key": "sk-SECRET", "seed": 7}})
    assert params["extra_body"]["seed"] == 7, "the structure around a secret must survive"
    assert "SECRET" not in str(params), params


def test_credentials_inside_lists_are_redacted(recording_handler):
    payload = recording_handler._redact({"headers": [{"authorization": "Bearer SECRET"}, {"accept": "json"}]})
    assert "SECRET" not in str(payload), payload
    assert payload["headers"][1]["accept"] == "json"


def test_verbose_metadata_is_redacted(recording_handler):
    """Verbose mode attaches raw callback kwargs -- the third path a credential can take."""
    from eqty_lineage.langchain import EqtyCallbackHandler

    verbose = EqtyCallbackHandler(verbose=True)
    out = verbose._verbose_metadata({"metadata": {"api_key": "sk-SECRET"}, "tags": ["prod"]})
    assert "SECRET" not in str(out), out
    assert recording_handler is not None


def test_redaction_keeps_the_key_visible(recording_handler):
    """A manifest should record that a credential was configured, not silently omit it."""
    kept = recording_handler._caller_metadata({"api_key": "sk-SECRET"})
    assert "api_key" in kept
    assert kept["api_key"] == recording_handler._REDACTED
