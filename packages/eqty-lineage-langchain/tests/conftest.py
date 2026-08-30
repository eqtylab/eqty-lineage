"""Fixtures for the LangChain handler tests.

The initialised SDK itself comes from the root ``conftest.py``: it has to be shared, because
``eqty_sdk.init()`` is process-global and a second call is ignored rather than refused.
"""

import pytest


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
