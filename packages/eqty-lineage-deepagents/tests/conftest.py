"""Fixtures for the DeepAgents handler tests.

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
    from eqty_lineage.deepagents import EqtyDeepAgentsHandler
    from eqty_lineage.langchain import EqtyCallbackHandler
    from eqty_sdk.context import graph_context

    class _Record(EqtyCallbackHandler):
        """Records each finalized computation.

        Mixed in *below* the DeepAgents handler rather than above it, so what it sees is what the
        handler actually recorded -- including the outputs the handler folds in on the way down, which
        is every file a tool wrote.
        """

        def _finalize(self, name, kind, input_cids, output_cids):
            self.computations.append((name, kind, [str(c) for c in input_cids], [str(c) for c in output_cids]))
            return super()._finalize(name, kind, input_cids, output_cids)

    class Recording(EqtyDeepAgentsHandler, _Record):
        def __init__(self) -> None:
            self.computations: list[tuple[str, str, list[str], list[str]]] = []
            super().__init__()

        def inputs_of(self, name: str) -> list[str]:
            return [i for n, _, ins, _ in self.computations if n == name for i in ins]

        def outputs_of(self, name: str) -> list[str]:
            return [o for n, _, _, outs in self.computations if n == name for o in outs]

        def kinds(self) -> set[str]:
            return {kind for _, kind, _, _ in self.computations}

    with graph_context(sdk):
        yield Recording()
