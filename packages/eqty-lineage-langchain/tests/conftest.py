"""Fixtures for the LangChain handler tests.

``eqty_sdk.init()`` is process-global and raises on a second call, so the SDK is initialised once per
session rather than per test. It writes a ``.eqty_sdk`` store relative to the working directory, which is
pointed at a temporary directory to keep runs from depositing state in the repository.
"""

import os

import pytest


@pytest.fixture(scope="session")
def sdk(tmp_path_factory):
    """One initialised SDK context for the whole run, or skip if the SDK is not installed."""
    eqty_sdk = pytest.importorskip("eqty_sdk", reason="eqty-sdk is not installed")

    store = tmp_path_factory.mktemp("eqty-store")
    previous = os.getcwd()
    os.chdir(store)
    try:
        ctx = eqty_sdk.Context.new("eqty-lineage-langchain tests")
        eqty_sdk.init(default_context=ctx).set_store_all_blobs(True)
        eqty_sdk.set_active_signer(eqty_sdk.Signer.new(name="eqty-lineage-langchain-tests", _load_if_exists=True))
        yield ctx
    finally:
        os.chdir(previous)


@pytest.fixture
def recording_handler(sdk):
    """A handler that records every computation it finalizes, bound to the session context.

    Asset constructors fall back to the active context but statement constructors do not, so the graph
    has to be built inside ``graph_context``.
    """
    from eqty_lineage.langchain import EqtyCallbackHandler
    from eqty_sdk.context import graph_context

    class Recording(EqtyCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self.computations: list[tuple[str, str, list[str], list[str]]] = []
            self.frameworks: list[str] = []

        def _finalize(self, name, kind, input_cids, output_cids):
            self.computations.append((name, kind, [str(c) for c in input_cids], [str(c) for c in output_cids]))
            self.frameworks.append(self._framework or "langchain")
            return super()._finalize(name, kind, input_cids, output_cids)

        def inputs_of(self, name: str) -> list[str]:
            return [i for n, _, ins, _ in self.computations if n == name for i in ins]

        def outputs_of(self, name: str) -> list[str]:
            return [o for n, _, _, outs in self.computations if n == name for o in outs]

    with graph_context(sdk):
        yield Recording()
